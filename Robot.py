import random as random

class Robot:
    def __init__(self, x, y, belief_map):
        # variables that changes
        self.x_loc = x
        self.y_loc = y

        # static variables
        self.start_loc = x, y
        self.SENSE_RANGE = 3.0 # Range for circles with bounds 41, 41
        # self.SENSE_RANGE = 7.0 # Range for circles with bounds 41, 41
        # self.SENSE_RANGE = 2.0 # Range for circles with bounds 41, 41
        self.belief_map = belief_map
        self.bounds = self.belief_map.get_bounds()
        self.sensor_model = None
        self.simulator = None

        self.exec_path = list()
        self.comm_exec_path = list()  # this is for communicate() with other robots

        # for oracle visualization
        r = random.random()
        b = random.random()
        g = random.random()
        self.color = (r, g, b)

    def reset_robot(self):
        self.x_loc = self.start_loc[0]
        self.y_loc = self.start_loc[1]
        self.exec_path = list()
    
    def move(self, direction):
        # move the robot while respecting bounds
        self.belief_map.is_valid_action(direction, self.get_loc(), self, update_state=True)

    def get_direction(self, curr_loc, next_loc):
        if (next_loc[0] - curr_loc[0] == -1):
            return 'left'
        elif (next_loc[0] - curr_loc[0] == 1):
            return 'right'
        elif (next_loc[1] - curr_loc[1] == 1):
            return 'backward'
        elif (next_loc[1] - curr_loc[1] == -1):
            return 'forward'

        return None

    def communicate_path(self, robots, curr_step, comm_step):
        if (curr_step % comm_step) == 0:
            for bot in robots:
                if bot == self:
                    continue
                bot.append_comm_exec_path(self.exec_path)

    # we just comm the occ and free locs
    def communicate_belief_map(self, robots, curr_step, comm_step):
        if (curr_step % comm_step) == 0:
            occupied_locs = self.belief_map.get_occupied_locs()
            free_locs = self.belief_map.get_free_locs()
            for bot in robots:
                if bot == self:
                    continue
                bot_belief_map = bot.get_belief_map()
                bot_belief_map.add_occupied_locs(occupied_locs)
                bot_belief_map.add_free_locs(free_locs)

    def get_bounds(self):
        return self.bounds

    def get_belief_map(self):
        return self.belief_map
    
    def get_sensor_model(self):
        return self.sensor_model

    def get_simulator(self):
        return self.simulator

    def get_color(self):
        return self.color

    def get_start_loc(self):
        return self.start_loc

    def get_loc(self):
        return (self.x_loc, self.y_loc)

    def get_exec_path(self):
        return self.exec_path

    def get_comm_exec_path(self):
        return self.comm_exec_path

    def get_sense_range(self):
        return self.SENSE_RANGE

    def append_exec_loc(self, loc):
        self.exec_path.append(loc)

    def append_comm_exec_path(self, exec_path):
        self.comm_exec_path += exec_path

    def change_xloc(self, x):
        self.x_loc += x

    def change_yloc(self, y):
        self.y_loc += y

    def set_loc(self, x_loc, y_loc):
        self.x_loc = x_loc
        self.y_loc = y_loc
    
    def set_color(self, color):
        self.color = color
    
    def set_belief_map(self, belief_map):
        self.belief_map = belief_map

    def set_sensor_model(self, sensor_model):
        self.sensor_model = sensor_model

    def set_simulator(self, simulator):
        self.simulator = simulator

