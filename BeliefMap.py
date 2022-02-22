from util import euclidean_distance

class BeliefMap:
    def __init__(self, sense_range, bounds):
        self.occupied_locs = set()
        self.free_locs = set()
        self.unknown_locs = set()

        for x in bounds:
            for y in bounds:
                self.unknown_locs.add(x, y)

        self.sense_range = sense_range

    def count_unknown_cells(self, bot_loc):
        scanned_unknown = set()

        for loc in self.unknown_locs:
           distance = euclidean_distance(bot_loc, loc)
           if (distance <= self.sense_range):
               scanned_unknown.add(loc)

        return scanned_unknown

    def update_map(self, gt_occupied_locs, gt_free_locs):
        for loc in gt_occupied_locs:
            if loc not in self.occupied_locs:
                self.occupied_locs.add(loc)
            if loc in self.unknown_locs:
                self.unknown_locs.remove(loc)

        for loc in gt_free_locs:
            if loc not in self.free_locs:
                self.free_locs.add(loc)
            if loc in self.unknown_locs:
                self.unknown_locs.remove(loc)



