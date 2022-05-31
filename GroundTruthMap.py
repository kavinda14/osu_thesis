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

    # def _get_occ_harbor(self):
    #     occ_locs = set()
        
    #     # top left pier
    #     length = int((3/4) * self.bounds[0]) #15
    #     width = int(self.bounds[1] / 8) #2
    #     for x in range(width):
    #         for y in range(length):
    #             occ_locs.add((x, y))

    #     # # bottom short pier
    #     # for x in range(15, 21):
    #     #     for y in range(4, 6):
    #     #         occ_locs.add((x, y))

    #     # middle mid-length pier
    #     for x in range(10, 21):
    #         for y in range(10, 12):
    #             occ_locs.add((x, y))

    #     for x in range (10, 12):
    #         for y in range(2, 12):
    #             occ_locs.add((x, y))

    #     # top long pier
    #     for x in range(5, 21):
    #         for y in range(19, 21):
    #             occ_locs.add((x, y))

    #     for x in range(2, 8):
    #         for y in range(7, 9):
    #             occ_locs.add((x, y))

        
    #     left_boat_locs = [(3, 1), (3, 4), (3, 10)]
    #     # left_boat_locs = [(3, 1), (3, 4), (3, 7), (3, 10)]
    #     # bottom_boat_locs = [(15, 1), (18, 1), (15, 7), (18, 7)]
    #     middle_boat_locs = [(18, 13), (15, 13), (12, 13)]
    #     top_boat_locs = [(18, 16), (15, 16), (12, 16), (9, 16), (6, 16)]

    #     # all_boat_locs = left_boat_locs + bottom_boat_locs + middle_boat_locs + top_boat_locs
    #     all_boat_locs = left_boat_locs + middle_boat_locs + top_boat_locs

    #     # randomize spawning of boats
    #     selected_boat_locs = set()
    #     for _ in range(len(all_boat_locs)):
    #         idx = np.random.randint(0, len(all_boat_locs)-1)
    #         selected_boat_locs.add(all_boat_locs[idx])

    #     # plot boat
    #     for loc in selected_boat_locs:
    #         x = loc[0]
    #         y = loc[1]
    #         occ_locs.add((x, y))
    #         occ_locs.add((x+1, y))
    #         occ_locs.add((x, y+1))
    #         occ_locs.add((x+1, y+1))


    #     return occ_locs

    # more complex T harbor
    def _get_occ_harbor(self):
        occ_locs = set()

        # middle mid-length pier
        for x in range(3, 21):
            for y in range(8, 10):
                occ_locs.add((x, y))
 
        # middle mid-length pier 2
        for x in range(3, 8):
            for y in range(15, 17):
                occ_locs.add((x, y))


        # middle piece
        for x in range(8, 10):
            for y in range(0, 19):
                occ_locs.add((x, y))



        # left_boat_locs = [(3, 1), (3, 4), (3, 7), (3, 10), (3, 13)]
        bottom_of_t_boat_locs = [(11, 5), (11, 2), (14, 5), (17, 5)]
        top_of_t_boat_locs = [(11, 11), (11, 14), (11, 17), (14, 11), (17, 11)]
        left_bottom_of_t = [(5, 5), (5, 2)]
        # bottom_boat_locs = [(15, 1), (18, 1), (15, 7), (18, 7)]
        # middle_boat_locs = [(18, 13), (15, 13), (12, 13), (12, 7)]
        # top_boat_locs = [(18, 16), (15, 16), (12, 16), (9, 16), (6, 16)]

        # all_boat_locs = left_boat_locs + bottom_boat_locs + middle_boat_locs + top_boat_locs
        # all_boat_locs = left_boat_locs + bottom_of_t_boat_locs + top_of_t_boat_locs
        all_boat_locs = bottom_of_t_boat_locs + top_of_t_boat_locs + left_bottom_of_t

        # randomize spawning of boats
        selected_boat_locs = set()
        for _ in range(len(all_boat_locs)):
            idx = np.random.randint(0, len(all_boat_locs)-1)
            selected_boat_locs.add(all_boat_locs[idx])

        # plot boat
        for loc in selected_boat_locs:
            x = loc[0]
            y = loc[1]
            occ_locs.add((x, y))
            occ_locs.add((x+1, y))
            occ_locs.add((x, y+1))
            occ_locs.add((x+1, y+1))

        return occ_locs

    # def _get_occ_harbor(self):
    #     occ_locs = set()

    #     # top left pier
    #     # length = int((3/4) * self.bounds[0]) #15
    #     # width = int(self.bounds[1] / 8) #2
    #     # for x in range(width):
    #     #     for y in range(length):
    #     #         occ_locs.add((x, y))

    #     # bottom short pier
    #     # for x in range(15, 21):
    #     #     for y in range(4, 6):
    #     #         occ_locs.add((x, y))

    #     # middle mid-length pier
    #     for x in range(8, 21):
    #         for y in range(8, 10):
    #             occ_locs.add((x, y))

    #     # middle piece
    #     for x in range(8, 10):
    #         for y in range(2, 19):
    #             occ_locs.add((x, y))

    #     # top long pier
    #     # for x in range(5, 21):
    #     #     for y in range(19, 21):
    #     #         occ_locs.add((x, y))
        
    #     # left_boat_locs = [(3, 1), (3, 4), (3, 7), (3, 10), (3, 13)]
    #     bottom_of_t_boat_locs = [(11, 5), (11, 2), (14, 5), (17, 5)]
    #     top_of_t_boat_locs = [(11, 11), (11, 14), (11, 17), (14, 11), (17, 11)]
    #     # bottom_boat_locs = [(15, 1), (18, 1), (15, 7), (18, 7)]
    #     # middle_boat_locs = [(18, 13), (15, 13), (12, 13), (12, 7)]
    #     # top_boat_locs = [(18, 16), (15, 16), (12, 16), (9, 16), (6, 16)]

    #     # all_boat_locs = left_boat_locs + bottom_boat_locs + middle_boat_locs + top_boat_locs
    #     # all_boat_locs = left_boat_locs + bottom_of_t_boat_locs + top_of_t_boat_locs
    #     all_boat_locs = bottom_of_t_boat_locs + top_of_t_boat_locs

    #     # randomize spawning of boats
    #     selected_boat_locs = set()
    #     for _ in range(len(all_boat_locs)):
    #         idx = np.random.randint(0, len(all_boat_locs)-1)
    #         selected_boat_locs.add(all_boat_locs[idx])

    #     # plot boat
    #     for loc in selected_boat_locs:
    #         x = loc[0]
    #         y = loc[1]
    #         occ_locs.add((x, y))
    #         occ_locs.add((x+1, y))
    #         occ_locs.add((x, y+1))
    #         occ_locs.add((x+1, y+1))


    #     return occ_locs


    # def _get_occ_harbor(self):
    #     occ_locs = set()
        
    #     # top left pier
    #     length = int((3/4) * self.bounds[0]) #15
    #     width = int(self.bounds[1] / 8) #2
    #     for x in range(width):
    #         for y in range(length):
    #             occ_locs.add((x, y))

    #     # bottom short pier
    #     for x in range(15, 21):
    #         for y in range(4, 6):
    #             occ_locs.add((x, y))

    #     # middle mid-length pier
    #     for x in range(10, 21):
    #         for y in range(10, 12):
    #             occ_locs.add((x, y))

    #     # top long pier
    #     for x in range(5, 21):
    #         for y in range(19, 21):
    #             occ_locs.add((x, y))
        
    #     left_boat_locs = [(3, 1), (3, 4), (3, 7), (3, 10)]
    #     bottom_boat_locs = [(15, 1), (18, 1), (15, 7), (18, 7)]
    #     middle_boat_locs = [(18, 13), (15, 13), (12, 13), (12, 7)]
    #     top_boat_locs = [(18, 16), (15, 16), (12, 16), (9, 16), (6, 16)]

    #     all_boat_locs = left_boat_locs + bottom_boat_locs + middle_boat_locs + top_boat_locs

    #     # randomize spawning of boats
    #     selected_boat_locs = set()
    #     for _ in range(len(all_boat_locs)):
    #         idx = np.random.randint(0, len(all_boat_locs)-1)
    #         selected_boat_locs.add(all_boat_locs[idx])

    #     # plot boat
    #     for loc in selected_boat_locs:
    #         x = loc[0]
    #         y = loc[1]
    #         occ_locs.add((x, y))
    #         occ_locs.add((x+1, y))
    #         occ_locs.add((x, y+1))
    #         occ_locs.add((x+1, y+1))


    #     return occ_locs
    
    
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

    def get_observation(self, bot, bot_loc):
        scanned_occupied = set()
        scanned_free = set()

        self._scan_locs(bot, bot_loc, self.occupied_locs, scanned_occupied)
        self._scan_locs(bot, bot_loc, self.free_locs, scanned_free)

        return [scanned_occupied, scanned_free]

    def _scan_locs(self, bot, bot_loc, exist_locs, scanned_list):
        sense_range = bot.get_sense_range()
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
