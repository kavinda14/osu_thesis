import math
import numpy as np

class SensorModel:
    def __init__(self, robot, world_map):
        self.robot = robot
        self.map = world_map
        self.sensing_range = robot.sensing_range

        self.final_partial_info = list()
        self.final_score = list()
        self.final_path = list()

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
        # Why is the +1 needed?
        path_matrix = np.zeros((bounds[0] + 1, bounds[1] + 1))

        for path in self.final_path:
            path_matrix[path] = 1

        print(path_matrix.astype(int))
        print("np size: ", np.size(path_matrix))

    # def final_partial_info_as_binary_matrix(self):
    #     bounds = self.map.get_bounds()

    #     path_matrix = np.empty((bounds[0], bounds[1]))

    #     for path in self.final_path:
    #         for x in path:
    #             for y in path:
    #                 if path[x, y] == 0:
    #                     path_matrix[x, y] = 1
    #                 else: 
    #                     path_matrix[x, y] = 0

    def append_score(self, score):
        self.final_score.append(score)

    def append_path(self, path):
        self.final_path.append(path)

    @staticmethod
    def euclidean_distance(p1, p2):
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]

        return math.sqrt((y2-y1)**2 + (x2-x1)**2)


