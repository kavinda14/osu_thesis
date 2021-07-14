import math
import sys
import numpy as np

class Map:
    def __init__(self, bounds, num_obstacles):
        """
        Inputs:
            robot: list of robot objects from Robot.py
        """
        #NOTE: These values scale the difficulty of the problem
        self.num_obstacles = num_obstacles
        self.bounds = bounds

        self.unobs_occupied = list()
        self.unobs_free = list()

        # Add obstacles to environment
        for i in range(self.num_obstacles):
            tetris_id = np.random.randint(0, 2)
            x = int(np.random.uniform(3, self.bounds[0] - 3, size=1))
            y = int(np.random.uniform(3, self.bounds[1] - 3, size=1))

            if tetris_id == 0: #Square
                self.unobs_occupied.append((x, y))
                self.unobs_occupied.append((x+1, y))
                self.unobs_occupied.append((x, y+1))
                self.unobs_occupied.append((x+1, y+1))
            else: #Straight line
                self.unobs_occupied.append((x, y))
                self.unobs_occupied.append((x+1, y))
                self.unobs_occupied.append((x+2, y))
                self.unobs_occupied.append((x+3, y))

        # Add free coords to unobs_free list
        for x in range(bounds[0]):
            for y in range(bounds[1]):
                if (x, y) not in self.unobs_occupied:
                    self.unobs_free.append((x, y))

    def scan(self, robot_loc, sensing_range):
        obs_obstacles = set()
        obs_free = set()
    
        for o_loc in self.unobs_occupied:
            distance = self.euclidean_distance(robot_loc, o_loc)
            if distance <= sensing_range:
                obs_obstacles.add(o_loc)
                self.unobs_occupied.remove(o_loc)
        for f_loc in self.unobs_free:
            distance = self.euclidean_distance(robot_loc, f_loc)
            if distance <= sensing_range:
                obs_free.add(f_loc)
                self.unobs_free.remote(f_loc)

        return [obs_obstacles, obs_free]

    @staticmethod
    def euclidean_distance(p1, p2):
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]

        return math.sqrt((y2-y1)**2 + (x2-x1)**2)