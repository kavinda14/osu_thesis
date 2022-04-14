import random as random

class Robot:
    def __init__(self, x, y, belief_map):
        # variables that changes
        self.x_loc = x
        self.y_loc = y

        # static variables
        self.start_loc = x, y
        # self.sensing_range = 2.85 # Range for square with bounds 21, 21
        # self.sensing_range = 4.3 # Range for circles with bounds 41, 41
        self.SENSE_RANGE = 3.0 # Range for circles with bounds 41, 41
        self.belief_map = belief_map
        self.bounds = self.belief_map.get_bounds()
        self.sensor_model = None
        self.simulator = None

        self.exec_paths = list()
        self.comm_exec_paths = list()  # this is for communicate() with other robots

        # for oracle visualization
        r = random.random()
        b = random.random()
        g = random.random()
        self.color = (r, g, b)

    def reset_robot(self):
        self.x_loc = self.start_loc[0]
        self.y_loc = self.start_loc[1]
        self.exec_paths = list()
    
    def move(self, direction):
        # move the robot while respecting bounds
        self.check_valid_move(direction, updateState=True)

    def get_direction(self, curr_loc, next_loc):
        if (next_loc[0] - curr_loc[0] == -1):
            return 'left'
        if (next_loc[0] - curr_loc[0] == 1):
            return 'right'
        if (next_loc[1] - curr_loc[1] == 1):
            return 'backward'
        if (next_loc[1] - curr_loc[1] == -1):
            return 'forward'

        return None

    def communicate_path(self, robots):
        for bot in robots:
            bot.set_comm_exec_paths(self.exec_paths)

    def communicate_belief_map(self, robots):
        occupied_locs = self.bot.get_belief_map().get_occupied_locs()
        for bot in robots:
            bot_belief_map = bot.get_belief_map()
            bot_belief_map.set_occupied_locs(occupied_locs)

    # def communicate(robots, obs_occupied_oracle, obs_free_oracle):
    # for bot1 in robots:
    #     sensor_model_bot1 = bot1.get_sensor_model()
    #     map_bot1 = bot1.get_map()
    #     other_paths = list()

    #     # for communicating the belief maps
    #     # by communicating these sets, the maps will contain these updates
    #     map_bot1.add_oracle_obs_free(obs_free_oracle)
    #     map_bot1.add_oracle_obs_occupied(obs_occupied_oracle)

    #     for bot2 in robots:
    #         if bot1 is not bot2:
    #             sensor_model_bot2 = bot2.get_sensor_model()
    #             final_path_bot2 = sensor_model_bot2.get_final_path()
    #             other_paths += final_path_bot2

    #     final_other_path_bot1 = sensor_model_bot1.get_final_other_path() + other_paths
    #     # final_other_path_bot1 = sensor_model_bot1.get_final_other_path().union(other_paths)
    #     sensor_model_bot1.set_final_other_path(final_other_path_bot1)

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

    def get_exec_paths(self):
        return self.exec_paths

    def get_comm_exec_paths(self):
        return self.comm_exec_paths

    def set_comm_exec_paths(self, exec_paths):
        self.comm_exec_paths += exec_paths

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

