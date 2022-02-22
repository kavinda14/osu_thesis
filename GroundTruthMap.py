import numpy as np
from util import euclidean_distance

class GroundTruthMap:
    def __init__(self, bounds, obs_density):

        #NOTE: These values scale the difficulty of the problem
        self.obs_density = obs_density
        self.bounds = bounds

        self.occupied_locs = set()
        self.free_locs = set()
       
    # This was written to select random starting locations for training
    def in_bounds(self, x_loc, y_loc):
        x = x_loc
        y = y_loc
        in_bounds = (x >= 0 and x < self.bounds[0] and y >= 0 and y < self.bounds[1])

        # Check unobs_occupied and obs_occupied from map
        for loc in self.occupied_locs:
            if x == loc[0] and y == loc[1]:
                return False

        for loc in self.obs_occupied:
            if x == loc[0] and y == loc[1]:
                return False

        return in_bounds

    def populate_obs_tetris(self):
        for _ in range(self.num_obstacles):
            tetris_id = np.random.randint(0, 2)
            x = int(np.random.uniform(3, self.bounds[0] - 3, size=1))
            y = int(np.random.uniform(3, self.bounds[1] - 3, size=1))

            if tetris_id == 0: # Square
                self.occupied_locs.add((x, y))
                self.occupied_locs.add((x+1, y))
                self.occupied_locs.add((x, y+1))
                self.occupied_locs.add((x+1, y+1))
            else: # Straight line
                self.occupied_locs.add((x, y))
                self.occupied_locs.add((x+1, y))
                self.occupied_locs.add((x+2, y))
                self.occupied_locs.add((x+3, y))

    def populate_obs_narrowrect(self):
        for _ in range(self.obs_density):
            rectangle_id = np.random.randint(0, 2)
            x = int(np.random.uniform(3, self.bounds[0] - 3, size=1))
            y = int(np.random.uniform(3, self.bounds[1] - 3, size=1))
            mid_point = (x, y)

            if rectangle_id == 0: # Horizontal rectangle
                self.occupied_locs.add(mid_point)
                self.occupied_locs.add((x+1, y))
                self.occupied_locs.add((x+2, y))
                self.occupied_locs.add((x-1, y))
            else: # Vertical rectangle
                self.occupied_locs.add(mid_point)
                self.occupied_locs.add((x, y+1))
                self.occupied_locs.add((x, y+2))
                self.occupied_locs.add((x, y-1))

    def populate_obs_circles(self):
        # circle at center
        x = self.bounds[0]//2
        y = self.bounds[1]//2
        self._create_obs(x, y, self.bounds)

        # circle at other locs
        for _ in range(self.obs_density):
            x = int(np.random.uniform(3, self.bounds[0] - 2, size=1))
            y = int(np.random.uniform(3, self.bounds[1] - 2, size=1))
            self._create_obs(x, y, self.bounds)
 
    def _create_obs(self, x, y, circle=True):
        mid_point = (x, y)
        self.occupied_locs.add(mid_point)

        for x in range(self.bounds[0]):
            for y in range(self.bounds[1]):
                distance = euclidean_distance((x, y), mid_point)
                if circle:
                    if distance < 2.3:  # 4.3 with map of 41, 41 bounds is good for circle
                        self.occupied_locs.add((x, y))
                else:
                    if distance < 1.6:  # 1.6 with map of 21, 21 bounds is good for square
                        self.occupied_locs.add((x, y))

    def populate_free(self):
        for x in range(self.bounds[0]):
            for y in range(self.bounds[1]):
                if (x, y) not in self.occupied_locs:
                    self.free_locs.add((x, y))

    def get_observation(self, bot_loc):
        scanned_occupied = set()
        scanned_free = set()

        self._scan_locs(bot_loc, self.occupied_locs, scanned_occupied)
        self._scan_locs(bot_loc, self.free_locs, scanned_free)

        return [scanned_occupied, scanned_free]

    def _scan_locs(self, bot_loc, exist_locs, scanned_list):
        for loc in exist_locs:
            distance = euclidean_distance(bot_loc, loc)
            if (distance <= self.sense_range):
                scanned_list.add(loc)

    def get_occupied_locs(self):
        return self.occupied_locs

    def get_free_locs(self):
        return self.free_locs

    def get_bounds(self):
        return self.bounds
