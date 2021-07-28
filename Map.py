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

        # The first two were kept as lists because I am iterating an modifying it at the same time in the scan function.
        self.unobs_occupied = set()
        self.unobs_free = set()
        # These two are only populated in the Simulator.
        self.obs_occupied = set()
        self.obs_free = set()

        # Add obstacles to environment
        for i in range(self.num_obstacles):
            tetris_id = np.random.randint(0, 2)
            x = int(np.random.uniform(3, self.bounds[0] - 3, size=1))
            y = int(np.random.uniform(3, self.bounds[1] - 3, size=1))

            if tetris_id == 0: # Square
                self.unobs_occupied.add((x, y))
                self.unobs_occupied.add((x+1, y))
                self.unobs_occupied.add((x, y+1))
                self.unobs_occupied.add((x+1, y+1))
            else: # Straight line
                self.unobs_occupied.add((x, y))
                self.unobs_occupied.add((x+1, y))
                self.unobs_occupied.add((x+2, y))
                self.unobs_occupied.add((x+3, y))

        # Add free coords to unobs_free list
        for x in range(bounds[0]):
            for y in range(bounds[1]):
                if (x, y) not in self.unobs_occupied:
                    self.unobs_free.add((x, y))

    def get_bounds(self):
        return self.bounds

    