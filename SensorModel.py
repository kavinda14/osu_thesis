import math
import numpy as np
from numpy.lib.function_base import disp

class SensorModel:
    def __init__(self, robot, world_map):
        self.robot = robot
        self.map = world_map
        self.sensing_range = robot.sensing_range

        self.final_partial_info = list()
        self.final_scores = list()
        self.final_path = list()
        self.final_other_path = list() # this is for communicate() with other robots
        self.final_path_matrices = list()
        self.final_actions = list()

    # update_map is there because in OraclePlanner, we use scan without updating the map
    # in Simulator with _update_map(), we use the update_map as True
    def scan(self, robot_loc, obs_occupied_oracle, update_map=True):
        scanned_obstacles = set()
        scanned_free = set()

        for o_loc in set(self.map.unobs_occupied):
            distance = self.euclidean_distance(robot_loc , o_loc)
            # the second part of the if statement was added for calculating unique reward
            if distance <= self.sensing_range and o_loc not in obs_occupied_oracle:
                scanned_obstacles.add(o_loc)
                if update_map:
                    self.map.unobs_occupied.remove(o_loc)

        for f_loc in set(self.map.unobs_free):
            distance = self.euclidean_distance(robot_loc, f_loc)
            if distance <= self.sensing_range:
                scanned_free.add(f_loc)
                if update_map:
                    self.map.unobs_free.remove(f_loc)

        return [scanned_obstacles, scanned_free]
    
    # this was created for mcts rollout as we are making a copy of the world map for simulations
    def scan_mcts(self, robot_loc, world_map):
        scanned_obstacles = set()
        scanned_free = set()

        for o_loc in set(world_map.unobs_occupied):
            distance = self.euclidean_distance(robot_loc , o_loc)
            if distance <= self.sensing_range:
                scanned_obstacles.add(o_loc)
                world_map.unobs_occupied.remove(o_loc)

        for f_loc in set(world_map.unobs_free):
            distance = self.euclidean_distance(robot_loc, f_loc)
            if distance <= self.sensing_range:
                scanned_free.add(f_loc)
                world_map.unobs_free.remove(f_loc)

        return [scanned_obstacles, scanned_free]

    # Called in Simulator
    # We keep update as true for getting the training data
    def create_partial_info(self, update=True):
        bounds = self.map.get_bounds()
        partial_info = np.empty((bounds[0], bounds[1]), dtype=int)

        for unobs_free_loc in self.map.unobs_free:
            partial_info[unobs_free_loc] = 2

        for unobs_occupied_loc in self.map.unobs_occupied:
            partial_info[unobs_occupied_loc] = 2

        for obs_free_loc in self.map.obs_free:
            partial_info[obs_free_loc] = 0

        for obs_occupied_loc in self.map.obs_occupied:
            partial_info[obs_occupied_loc] = 1
        
        if update:
            self.final_partial_info.append(partial_info)
        else: 
            return partial_info

    # the diff here and the other same function is that we are passing the map we want into this
    def create_partial_info_mcts(self, map, update=True):
        bounds = map.get_bounds()
        partial_info = np.empty((bounds[0], bounds[1]), dtype=int)

        for unobs_free_loc in map.unobs_free:
            partial_info[unobs_free_loc] = 2

        for unobs_occupied_loc in map.unobs_occupied:
            partial_info[unobs_occupied_loc] = 2

        for obs_free_loc in map.obs_free:
            partial_info[obs_free_loc] = 0

        for obs_occupied_loc in map.obs_occupied:
            partial_info[obs_occupied_loc] = 1
        
        if update:
            self.final_partial_info.append(partial_info)
        else: 
            return partial_info
    
    # update flag was added because when running greedy planner with NN, we want to get path but not update final list
    def create_final_path_matrix(self, update=True):
        bounds = self.map.get_bounds()
        path_matrix = np.zeros((bounds[0], bounds[1]), dtype=int)
        debug_matrix = np.zeros((bounds[0], bounds[1]), dtype=int)

        for path in self.final_path:
            path_matrix[path] = 1
            debug_matrix[path] = 1
        
        # this is for multi-robot when communication of other_paths is done
        for path in self.final_other_path:
            path_matrix[path] = 1

        if update:
            self.final_path_matrices.append(path_matrix)

        else:
            return path_matrix

    def create_final_path_matrix_mcts(self, input_final_path_matrix, update=True):
        bounds = self.map.get_bounds()
        path_matrix = np.zeros((bounds[0], bounds[1]), dtype=int)

        for path in input_final_path_matrix:
            path_matrix[path] = 1

        if update:
            self.final_path_matrices.append(path_matrix)
        else:
            return path_matrix

    def create_binary_matrices(self, input_list):
        binary_matrices = list()
        
        for main_matrix in input_list:

            matrix_list = list()
            n = 1

            while (n <= 3):
                sub_matrix = np.empty((np.shape(main_matrix)), dtype=int)
                for x in range(np.shape(main_matrix)[0]):
                    for y in range(np.shape(main_matrix)[1]):
                        # obs_free
                        if n == 1:
                            if main_matrix[x, y] == 0:
                                sub_matrix[x, y] = 1
                            else:
                                sub_matrix[x, y] = 0

                        # obs_occupied
                        if n == 2:
                            if main_matrix[x, y] == 1:
                                sub_matrix[x, y] = 1
                            else:
                                sub_matrix[x, y] = 0

                        # unobs
                        if n == 3:
                            if main_matrix[x, y] == 2:
                                sub_matrix[x, y] = 1
                            else:
                                sub_matrix[x, y] = 0
                
                matrix_list.append(sub_matrix)
                n += 1

            binary_matrices.append(matrix_list)
        
        return binary_matrices

    # greedy_planner flag is added because we need to return action matrix
    # *args was added because in mcts network_reward, we need some way of passing the the robot location
    def create_action_matrix(self, action, greedy_planner=False):
        # Think of this as an action but a diff way of representing it
        # This function needs to be called before we move the robot in the Simulator

        # Create empty matrix of same bounds
        # Fill matrix in with the obs_occupied digit = 1
        # Get the action location 
        # Get the mid-point of the matrix = [x/2, y/2]
        # Get displacement = mid-point - action_loc
        # Iterate through all the values of matrix1
            # If coord + displacement is within bounds:
                # matrix2[coord + displacement] = matrix1[coord]

        bounds = self.map.get_bounds()
        action_matrix = np.ones((bounds[0], bounds[1]), dtype=int)
        mid_point = [bounds[0]//2, bounds[1]//2]
        # Assumption is made that the action is valid
        action_loc = self.robot.get_action_loc(action)

        displacement = [(mid_point[0] - action_loc[0]), (mid_point[1] - action_loc[1])]

        partial_info = self.final_partial_info[-1]

        for x in range(len(partial_info)):
            if (x + displacement[0]) < bounds[0]:
                for y in range(len(partial_info)):
                    if (y + displacement[1] < bounds[1]):
                        action_matrix[(x + displacement[0]), (y + displacement[1])] = partial_info[x, y]

        """"
        [[2 2 2 2 2]
        [2 0 0 0 2]
        [2 0 0 0 2]
        [2 0 0 0 2]
        [2 2 2 2 2]]

        action = forward

        [[1 2 2 2 2]
        [1 2 0 0 0]
        [1 2 0 0 0]
        [1 2 0 0 0]
        [1 2 2 2 2]]

        Remember that the x axis here is the left corner going downwards.
        Y axis is going to the right.
        So an action of forward where (y-1) means that the action will be to the left of robot mid-point from my frame.
        """
        if greedy_planner:
            return action_matrix
            
        # refractor: this should not be appending to a final list. 
        # this should just return the action matrix and the appending should happen elsewhere
        self.final_actions.append(action_matrix)

    def create_action_matrix_mcts(self, action_loc):
        # Think of this as an action but a diff way of representing it
        # This function needs to be called before we move the robot in the Simulator

        # Create empty matrix of same bounds
        # Fill matrix in with the obs_occupied digit = 1
        # Get the action location 
        # Get the mid-point of the matrix = [x/2, y/2]
        # Get displacement = mid-point - action_loc
        # Iterate through all the values of matrix1
            # If coord + displacement is within bounds:
                # matrix2[coord + displacement] = matrix1[coord]

        bounds = self.map.get_bounds()
        action_matrix = np.ones((bounds[0], bounds[1]), dtype=int)
        mid_point = [bounds[0]//2, bounds[1]//2]
        # Assumption is made that the action is valid
    
        displacement = [(mid_point[0] - action_loc[0]), (mid_point[1] - action_loc[1])]

        partial_info = self.final_partial_info[-1]

        for x in range(len(partial_info)):
            if (x + displacement[0]) < bounds[0]:
                for y in range(len(partial_info)):
                    if (y + displacement[1] < bounds[1]):
                        action_matrix[(x + displacement[0]), (y + displacement[1])] = partial_info[x, y]

        """"
        [[2 2 2 2 2]
        [2 0 0 0 2]
        [2 0 0 0 2]
        [2 0 0 0 2]
        [2 2 2 2 2]]

        action = forward

        [[1 2 2 2 2]
        [1 2 0 0 0]
        [1 2 0 0 0]
        [1 2 0 0 0]
        [1 2 2 2 2]]

        Remember that the x axis here is the left corner going downwards.
        Y axis is going to the right.
        So an action of forward where (y-1) means that the action will be to the left of robot mid-point from my frame.
        """
        return action_matrix
            
    def append_action_matrix(self, matrix):
        self.final_actions.append(matrix)

    def append_score(self, score):
        self.final_scores.append(score)

    def append_path(self, path):
        self.final_path.append(path)

    def get_final_path(self):
        return self.final_path

    def get_final_other_path(self):
        return self.final_other_path

    def get_final_partial_info(self):
        return self.final_partial_info

    def get_final_actions(self):
        return self.final_actions

    def get_final_scores(self):
        return self.final_scores

    def get_final_path_matrices(self):
        return self.final_path_matrices

    def set_final_path(self, paths):
        self.final_path = paths

    def set_final_other_path(self, final_other_path):
        self.final_other_path = final_other_path

    @staticmethod
    def euclidean_distance(p1, p2):
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]

        return math.sqrt((y2-y1)**2 + (x2-x1)**2)




