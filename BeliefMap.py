from utils import euclidean_distance

class BeliefMap:
    def __init__(self, bounds):
        self.occupied_locs = set()
        self.free_locs = set()
        self.unknown_locs = set()
        self.bounds = bounds

        # at step=0, the full map is unknown
        for x in range(self.bounds[0]):
            for y in range(self.bounds[1]):
                self.unknown_locs.add((x, y))

    def count_unknown_cells(self, sense_range, bot_loc):
        scanned_unknown = set()
        for loc in self.unknown_locs:
           distance = euclidean_distance(bot_loc, loc)
           if (distance <= sense_range):
               scanned_unknown.add(loc)

        return scanned_unknown

    # call get_observation() from GroundTruthMap.py before calling this in Simulator.py 
    def update_map(self, obs_occupied_locs, obs_free_locs):
        # add occ locs to belief map
        for loc in obs_occupied_locs:
            if loc not in self.occupied_locs:
                self.occupied_locs.add(loc)
            if loc in self.unknown_locs:
                self.unknown_locs.remove(loc)

        # add free locs to belief map
        for loc in obs_free_locs:
            if loc not in self.free_locs:
                self.free_locs.add(loc)
            if loc in self.unknown_locs:
                self.unknown_locs.remove(loc)

    def is_valid_action(self, direction, bot_curr_loc, bot=None, update_state=False):
        x_loc = bot_curr_loc[0]
        y_loc = bot_curr_loc[1]

        if direction == 'left':
            valid = self.is_valid_loc(x_loc-1, y_loc)
            if valid and update_state:
                bot.change_xloc(-1)

        elif direction == 'right':
            valid = self.is_valid_loc(x_loc+1, y_loc)
            if valid and update_state:
                bot.change_xloc(+1)

        elif direction == 'backward':
            valid = self.is_valid_loc(x_loc, y_loc+1)
            if valid and update_state:
                bot.change_yloc(+1)

        elif direction == 'forward':
            valid = self.is_valid_loc(x_loc, y_loc-1)
            if valid and update_state:
                bot.change_yloc(-1)
        else:
            raise ValueError(f"Robot received invalid direction: {direction}!")

        return valid

    def is_valid_loc(self, x_loc, y_loc):
        in_bounds = (x_loc >= 0 and x_loc <
                     self.bounds[0] and y_loc >= 0 and y_loc < self.bounds[1])
        
        if in_bounds:
            for loc in self.free_locs:
                if x_loc == loc[0] and y_loc == loc[1]:
                    return True
        
        return False

    def get_action_loc(self, action, curr_bot_loc):
        if action == 'left':
            action_loc = [curr_bot_loc[0]-1, curr_bot_loc[1]]

        elif action == 'right':
            action_loc = [curr_bot_loc[0]+1, curr_bot_loc[1]]

        elif action == 'backward':
            action_loc = [curr_bot_loc[0], curr_bot_loc[1]+1]

        elif action == 'forward':
            action_loc = [curr_bot_loc[0], curr_bot_loc[1]-1]

        # this was added for the mcts reward function
        elif action == 'root':
            action_loc = curr_bot_loc

        return tuple(action_loc)

    def get_bounds(self):
        return self.bounds

    def get_unknown_locs(self):
        return self.unknown_locs

    def get_occupied_locs(self):
        return self.occupied_locs
    
    def get_free_locs(self):
        return self.free_locs

    def get_name(self):
        return self.name

    def add_occupied_locs(self, occupied_locs):
        self.occupied_locs = self.occupied_locs.union(occupied_locs)

    def add_free_locs(self, free_locs):
        self.free_locs = self.free_locs.union(free_locs)

