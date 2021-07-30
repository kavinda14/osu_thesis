import math
import numpy as np
from numpy.lib.function_base import disp

class SensorModel:
    def __init__(self, robot, world_map):
        self.robot = robot
        self.map = world_map
        self.sensing_range = robot.sensing_range

        self.final_partial_info = list()
        self.final_score = list()
        self.final_path = list()
        self.final_actions = list()

    def scan(self):
        scanned_obstacles = set()
        scanned_free = set()
        robot_loc = self.robot.get_loc()

        for o_loc in set(self.map.unobs_occupied):
            distance = self.euclidean_distance(robot_loc , o_loc)
            if distance <= self.sensing_range:
                scanned_obstacles.add(o_loc)
                self.map.unobs_occupied.remove(o_loc)

        for f_loc in set(self.map.unobs_free):
            distance = self.euclidean_distance(robot_loc, f_loc)
            if distance <= self.sensing_range:
                scanned_free.add(f_loc)
                self.map.unobs_free.remove(f_loc)

        return [scanned_obstacles, scanned_free]

    # Called in Simulator
    def create_partial_info(self):
        bounds = self.map.get_bounds()
        partial_info = np.empty((bounds[0], bounds[1]))

        for obs_free_loc in self.map.obs_free:
            partial_info[obs_free_loc] = 0

        for obs_occupied_loc in self.map.obs_occupied:
            partial_info[obs_occupied_loc] = 1

        for unobs_free_loc in self.map.unobs_free:
            partial_info[unobs_free_loc] = 2

        for unobs_occupied_loc in self.map.unobs_occupied:
            partial_info[unobs_occupied_loc] = 2
        
        self.final_partial_info.append(partial_info.astype(int))

    def final_path_as_matrix(self):
        bounds = self.map.get_bounds()
        path_matrix = np.zeros((bounds[0], bounds[1]))

        for path in self.final_path:
            path_matrix[path] = 1

        print(path_matrix.astype(int))
        print("np size: ", np.shape(path_matrix))

    def final_partial_info_as_binary_matrices(self):
        final_partial_info_binary_matrices = list()
        
        for partial_info in self.final_partial_info:

            matrix_list = list()
            n = 1

            while (n <= 3):
                matrix = np.empty((np.shape(partial_info)))
                for x in range(np.shape(partial_info)[0]):
                    for y in range(np.shape(partial_info)[1]):
                        # obs_free
                        if n == 1:
                            if partial_info[x, y] == 0:
                                matrix[x, y] = 1
                            else:
                                matrix[x, y] = 0

                        # obs_occupied
                        if n == 2:
                            if partial_info[x, y] == 1:
                                matrix[x, y] = 1
                            else:
                                matrix[x, y] = 0

                        # unobs
                        if n == 3:
                            if partial_info[x, y] == 2:
                                matrix[x, y] = 1
                            else:
                                matrix[x, y] = 0
                
                matrix_list.append(matrix.astype(int))
                n += 1

            final_partial_info_binary_matrices.append(matrix_list)
        
        return final_partial_info_binary_matrices

    def create_action_matrix(self, action):
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
        actions_matrix = np.ones((bounds[0], bounds[1]), dtype=int)
        robot_loc = self.robot.get_loc()
        action_loc = []
        mid_point = [bounds[0]//2, bounds[1]//2]

        # Assumption is made that the action is valid
        if action == 'left':
            action_loc = [robot_loc[0]-1, robot_loc[1]]

        elif action == 'right':
            action_loc = [robot_loc[0]+1, robot_loc[1]]
        
        elif action == 'backward':
            action_loc = [robot_loc[0], robot_loc[1]+1]

        elif action == 'forward':
            action_loc = [robot_loc[0]-1, robot_loc[1]-1]

        displacement = [(mid_point[0] - action_loc[0]), (mid_point[1] - action_loc[1])]

        partial_info = self.final_partial_info[-1]

        for x in range(len(partial_info)):
            if (x + displacement[0]) < bounds[0]:
                for y in range(len(partial_info)):
                    if (y + displacement[1] < bounds[1]):
                        actions_matrix[(x + displacement[0]), (y + displacement[1])] = partial_info[x, y]
        
        print(action_loc)
        print("ACTION MATRIX: ", actions_matrix)

    def append_action_matrix(self, matrix):
        self.final_actions.append(matrix)

    def append_score(self, score):
        self.final_score.append(score)

    def append_path(self, path):
        self.final_path.append(path)

    def get_final_partial_info(self):
        return self.final_partial_info

    @staticmethod
    def euclidean_distance(p1, p2):
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]

        return math.sqrt((y2-y1)**2 + (x2-x1)**2)


# if __name__ == "__main__":

#    mid_point = [2, 2]
#    action_loc = [1, 1]
   
#    displacement = mid_point - action_loc
#    print(displacement)


