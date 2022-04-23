import numpy as np

class SensorModel:
    def __init__(self, bot, belief_map):
        self.bot = bot
        self.belief_map = belief_map

        self.partial_info_matrices = list()
        self.path_matrices = list()
        self.comm_path_matrices = list()
        self.action_matrices = list()
    
    # this was created for mcts rollout as we are making a copy of the world map for simulations
    def scan_mcts(self, robot_loc, unobs_free, unobs_occupied):
        scanned_obstacles = set()
        scanned_free = set()

        for o_loc in set(unobs_occupied):
            distance = self.euclidean_distance(robot_loc , o_loc)
            if distance <= self.sensing_range:
                scanned_obstacles.add(o_loc)
                unobs_occupied.remove(o_loc)

        for f_loc in set(unobs_free):
            distance = self.euclidean_distance(robot_loc, f_loc)
            if distance <= self.sensing_range:
                scanned_free.add(f_loc)
                unobs_free.remove(f_loc)

        return [scanned_obstacles, scanned_free]

    # Called in Simulator
    # We keep update as true for getting the training data
    def create_partial_info(self, update=True):
        bounds = self.belief_map.get_bounds()
        partial_info = np.empty((bounds[0], bounds[1]), dtype=int)

        unknown_locs = self.belief_map.get_unknown_locs()
        for loc in unknown_locs:
            partial_info[loc] = 2

        free_locs = self.belief_map.get_free_locs()
        for loc in free_locs:
            partial_info[loc] = 0

        occupied_locs = self.belief_map.get_occupied_locs()
        for loc in occupied_locs:
            partial_info[loc] = 1
        
        if update:
            self.partial_info_matrices.append(partial_info)
        else: 
            return partial_info

    # the diff here and the other same function is that we are passing the map objects we want into this
    def create_partial_info_mcts(self, unobs_free, unobs_occupied, obs_occupied, obs_free, bounds, update=True):
        partial_info = np.empty((bounds[0], bounds[1]), dtype=int)

        for unobs_free_loc in unobs_free:
            partial_info[unobs_free_loc] = 2

        for unobs_occupied_loc in unobs_occupied:
            partial_info[unobs_occupied_loc] = 2

        for obs_free_loc in obs_free:
            partial_info[obs_free_loc] = 0

        for obs_occupied_loc in obs_occupied:
            partial_info[obs_occupied_loc] = 1

        if update:
            self.partial_info_matrices.append(partial_info)
        else:
            return partial_info
    
    # update flag was added because when running greedy planner with NN, we want to get path but not update final list
    def create_path_matrix(self, update=True):
        bounds = self.belief_map.get_bounds()
        path_matrix = np.zeros((bounds[0], bounds[1]), dtype=int)

        exec_paths = self.bot.get_exec_path()
        for path in exec_paths:
            path_matrix[path] = 1
        
        # this is for multi-robot when communication of other_paths is done
        comm_exec_paths = self.bot.get_comm_exec_path()
        for path in comm_exec_paths:
            path_matrix[path] = 1

        if update:
            self.path_matrices.append(path_matrix)

        else:
            return path_matrix
    
    # this was created to combine path and others together for multi-robot rollout
    # it differs from create_path_matrix() because it adds ONLY the exec_paths and NOT comm_exec_paths
    def create_rollout_path_matrix(self, update=True):
        bounds = self.belief_map.get_bounds()
        path_matrix = np.zeros((bounds[0], bounds[1]), dtype=int)

        exec_paths = self.bot.get_exec_path()
        for path in exec_paths:
            path_matrix[path] = 1
        
        if update:
            self.path_matrices.append(path_matrix)

        else:
            return path_matrix

    # this was created to combine path and others together for multi-robot rollout
    def create_rollout_comm_path_matrix(self, update=True):
        bounds = self.belief_map.get_bounds()
        path_matrix = np.zeros((bounds[0], bounds[1]), dtype=int)

        comm_exec_paths = self.bot.get_comm_exec_path()
        for path in comm_exec_paths:
            path_matrix[path] = 1

        if update:
            self.comm_path_matrices.append(path_matrix)

        else:
            return path_matrix

    def create_final_path_matrix_mcts(self, input_final_path_matrix, update=True):
        bounds = self.belief_map.get_bounds()
        path_matrix = np.zeros((bounds[0], bounds[1]), dtype=int)

        for path in input_final_path_matrix:
            path_matrix[path] = 1

        # this is for multi-robot when communication of other_paths is done
        comm_exec_paths = self.bot.get_comm_exec_path()
        for path in comm_exec_paths:
            path_matrix[path] = 1

        if update:
            self.path_matrices.append(path_matrix)
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
    def create_action_matrix(self, action, curr_bot_loc, greedy_planner=False):
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

        bounds = self.belief_map.get_bounds()
        action_matrix = np.ones((bounds[0], bounds[1]), dtype=int)
        mid_point = [bounds[0]//2, bounds[1]//2]
        # assumption is made that the action is valid
        action_loc = self.belief_map.get_action_loc(action, curr_bot_loc)

        displacement = [(mid_point[0] - action_loc[0]), (mid_point[1] - action_loc[1])]

        partial_info = self.partial_info_matrices[-1]

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
        self.action_matrices.append(action_matrix)

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

        bounds = self.belief_map.get_bounds()
        action_matrix = np.ones((bounds[0], bounds[1]), dtype=int)
        mid_point = [bounds[0]//2, bounds[1]//2]
        # Assumption is made that the action is valid
    
        displacement = [(mid_point[0] - action_loc[0]), (mid_point[1] - action_loc[1])]

        partial_info = self.partial_info_matrices[-1]

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
        self.action_matrices.append(matrix)

    def get_partial_info_matrices(self):
        return self.partial_info_matrices

    def get_action_matrices(self):
        return self.action_matrices

    def get_path_matrices(self):
        return self.path_matrices
    
    def get_comm_path_matrices(self):
        return self.comm_path_matrices



