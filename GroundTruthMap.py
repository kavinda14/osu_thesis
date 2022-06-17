import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import euclidean_distance

class GroundTruthMap:
    def __init__(self, bounds, OCC_DENSITY):
        #NOTE: These values scale the difficulty of the problem
        self.OCC_DENSITY = OCC_DENSITY
        self.bounds = bounds

        self.occupied_locs = self._get_occ_harbor()
        self.free_locs = self._get_free()

    def _get_occ_harbor(self):
        occ_locs = set()
        
        self._get_circular_dock((11, 34), occ_locs) # top circular dock
        self._get_circular_dock((11, 14), occ_locs) # bottom circular dock
        
        # vertical pier
        for x in range(30, 33):
            for y in range(8, 41):
                occ_locs.add((x, y))

        boat_locs = [(35, 35), (35, 29), (35, 23), (35, 17), (35, 11)]
        # randomly select boat locs
        selected_boat_locs = set()
        for _ in range(len(boat_locs)):
            idx = np.random.randint(0, len(boat_locs))
            selected_boat_locs.add(boat_locs[idx])

        # plot boats
        for loc in selected_boat_locs:
            x_init = loc[0]
            y_init = loc[1]
            for x in range(x_init, x_init+4):
                for y in range(y_init, y_init+4):
                    occ_locs.add((x, y))

        return occ_locs

    def _get_circular_dock(self, mid_loc, occ_locs):
        x_mid = mid_loc[0]
        y_mid = mid_loc[1]
        
        # circular dock
        for x in range(x_mid-2, x_mid+2):
            for y in range(y_mid-6, y_mid-2):
                occ_locs.add((x, y))

        # vertical pole
        for x in range(x_mid-1, x_mid+1):
            for y in range(y_mid-12, y_mid+4):
                occ_locs.add((x, y))

        # horizontal pole
        for x in range(x_mid-8, x_mid+8):
            for y in range(y_mid-5, y_mid-3):
                occ_locs.add((x, y))

        boat_locs = list()
        boat_locs.append((x_mid-7, y_mid-1))  # top left
        boat_locs.append((x_mid+3, y_mid-1))  # top right
        boat_locs.append((x_mid-7, y_mid-11))  # bottom left
        boat_locs.append((x_mid+3, y_mid-11))  # bottom right

        # randomly select boat locs
        selected_boat_locs = set()
        for _ in range(len(boat_locs)):
            idx = np.random.randint(0, len(boat_locs))
            selected_boat_locs.add(boat_locs[idx])

        # plot boats
        for loc in selected_boat_locs:
            x_init = loc[0]
            y_init = loc[1]
            for x in range(x_init, x_init+4):
                for y in range(y_init, y_init+4):
                    occ_locs.add((x, y))

    
    def _get_occ_tetris(self):
        occ_locs = set()
        for _ in range(self.OCC_DENSITY):
            tetris_id = np.random.randint(0, 2)
            x = int(np.random.uniform(3, self.bounds[0] - 3, size=1))
            y = int(np.random.uniform(3, self.bounds[1] - 3, size=1))

            if tetris_id == 0: # Square
                occ_locs.add((x, y))
                occ_locs.add((x+1, y))
                occ_locs.add((x, y+1))
                occ_locs.add((x+1, y+1))
            else: # Straight line
                occ_locs.add((x, y))
                occ_locs.add((x+1, y))
                occ_locs.add((x+2, y))
                occ_locs.add((x+3, y))
        
        return occ_locs

    def _get_occ_narrowrect(self):
        occ_locs = set()
        for _ in range(self.OCC_DENSITY):
            rectangle_id = np.random.randint(0, 2)
            x = int(np.random.uniform(3, self.bounds[0] - 3, size=1))
            y = int(np.random.uniform(3, self.bounds[1] - 3, size=1))
            mid_point = (x, y)

            if rectangle_id == 0: # Horizontal rectangle
                occ_locs.add(mid_point)
                occ_locs.add((x+1, y))
                occ_locs.add((x+2, y))
                occ_locs.add((x-1, y))
            else: # Vertical rectangle
                occ_locs.add(mid_point)
                occ_locs.add((x, y+1))
                occ_locs.add((x, y+2))
                occ_locs.add((x, y-1))

        return occ_locs

    def _get_occ_circles(self):
        occ_locs = set()
        # circle at center
        x = self.bounds[0]//2
        y = self.bounds[1]//2
        self._create_obj(x, y, occ_locs)

        # np.random.seed(8)  # this can be used for debugging when we want the same kind of map

        # circle at other locs
        for _ in range(self.OCC_DENSITY):
            x = int(np.random.uniform(3, self.bounds[0] - 2, size=1))
            y = int(np.random.uniform(3, self.bounds[1] - 2, size=1))
            self._create_obj(x, y, occ_locs)

        return occ_locs
 
    def _create_obj(self, x, y, occ_locs, circle=True):
        mid_point = (x, y)
        occ_locs.add(mid_point)

        for x in range(self.bounds[0]):
            for y in range(self.bounds[1]):
                distance = euclidean_distance((x, y), mid_point)
                if circle:
                    if distance < 2.3:  # 4.3 with map of 41, 41 bounds is good for circle
                        occ_locs.add((x, y))
                else:
                    if distance < 1.6:  # 1.6 with map of 21, 21 bounds is good for square
                        occ_locs.add((x, y))

    def _get_free(self):
        free_locs = set()
        for x in range(self.bounds[0]):
            for y in range(self.bounds[1]):
                if (x, y) not in self.occupied_locs:
                    free_locs.add((x, y))

        return free_locs

    def is_valid_loc(self, bot_loc):
        x_loc = bot_loc[0]
        y_loc = bot_loc[1]
        in_bounds = (x_loc >= 0 and x_loc <
                     self.bounds[0] and y_loc >= 0 and y_loc < self.bounds[1])

        for loc in self.occupied_locs:
            if x_loc == loc[0] and y_loc == loc[1]:
                return False

        return in_bounds

    # is_oracle added because we want ONLY the oracle planner to have larger scanning radius
    def get_observation(self, bot_loc, sense_range):
        scanned_occupied = set()
        scanned_free = set()

        self._scan_locs(bot_loc, self.occupied_locs, scanned_occupied, sense_range)
        self._scan_locs(bot_loc, self.free_locs, scanned_free, sense_range)

        return [scanned_occupied, scanned_free]

    def _scan_locs(self, bot_loc, exist_locs, scanned_list, sense_range):
        for loc in exist_locs:
            distance = euclidean_distance(bot_loc, loc)
            if (distance <= sense_range):
                scanned_list.add(loc)

    def visualize(self):
        plt.xlim(0, self.bounds[0])
        plt.ylim(0, self.bounds[1])

        ax = plt.gca()
        ax.set_aspect('equal', 'box')

        for spot in self.free_locs:
            hole = patches.Rectangle(spot, 1, 1, facecolor='white')
            ax.add_patch(hole)

        for spot in self.occupied_locs:
            hole = patches.Rectangle(spot, 1, 1, facecolor='green')
            ax.add_patch(hole)

        plt.show()

    def get_occupied_locs(self):
        return self.occupied_locs

    def get_free_locs(self):
        return self.free_locs

    def get_bounds(self):
        return self.bounds

    def get_name(self):
        return self.name
