from random import randint
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
sys.path.insert(0, './basic_MCTS_python')
from copy import deepcopy


class Simulator:
    def __init__(self, belief_map, ground_truth_map, bot, sensor_model, generate_data):
        self.ground_truth_map = ground_truth_map
        self.belief_map = belief_map
        self.bot = bot
        self.sensor_model = sensor_model

        self.curr_score = 0
        self.scores = list()

        # Simulator.py is used for eval and data generation
        self.generate_data = generate_data

        self.sys_actions = ['left', 'right', 'forward', 'backward']

        self.debug_network_score = list()
        self.debug_greedy_score = list()

    # creates the initially matrices needed
    def _initialize_data_matrices(self):
        self.sensor_model.create_partial_info()
        self.scores.append(self.curr_score)  # init score is 0

        # at the start, there is no action, so we just add the initial partial info into the action matrix list
        partial_info_matrix = self.sensor_model.get_partial_info_matrices()[0]
        self.sensor_model.append_action_matrix(partial_info_matrix)

        # to initialize a matrix in comm_path_matrices for data generation ONLY
        curr_bot_loc = self.bot.get_loc()
        # keep in mind that for rollout in data generation, we create the path matrices separately and then combine them
        self.sensor_model.create_rollout_path_matrix()
        self.sensor_model.create_rollout_comm_path_matrix()

        path_matrix = self.sensor_model.get_path_matrices()[0]
        path_matrix[curr_bot_loc[0]][curr_bot_loc[1]] = 1

        path_matrix = self.sensor_model.get_comm_path_matrices()[0]
    

    def _generate_data_matrices(self, action):
        self.sensor_model.create_partial_info()
        self.sensor_model.create_rollout_path_matrix()
        # self.sensor_model.create_path_matrix()
        self.sensor_model.create_rollout_comm_path_matrix()
        self.sensor_model.create_action_matrix(action, self.bot.get_loc())


    # train is there because of the backtracking condition in each planner 
    def run(self, planner, robot_occupied_locs, curr_step, robot_curr_locs):       
        # on step=0, we just initialize map and matrices
        if curr_step == 0:
            self._initialize_data_matrices()
            return

        # get action from planner 
        action = planner.get_action(self.bot)

        # to make sure that robots aren't in the same loc
        # only mcts is allowed this because other planners have backtrack count and mcts does not - this makes it fair
        if planner.__class__.__name__ == "MCTS":
            if (self.belief_map.get_action_loc(action, self.bot.get_loc())) in robot_curr_locs:
                while True:
                    old_action = action
                    idx = randint(0, len(self.sys_actions)-1)
                    action = self.sys_actions[idx]
                    new_loc = self.belief_map.get_action_loc(action, self.bot.get_loc())
                    if self.belief_map.is_valid_loc(new_loc[0], new_loc[1]) and action is not old_action:
                        break

        self._generate_data_matrices(action) # must be called before moving - we use curr info of where we are along with action to predict what the score would be
        
        # remember that if we call move() at curr_step=0 and 1, all actions will return False because the BeliefMap has no free_locs for is_valid_loc()
        self.bot.move(action)

        # sanity check the robot is in bounds after moving
        if not self.ground_truth_map.is_valid_loc(self.bot.get_loc()):
            raise ValueError(
                f"Robot has left the map. It is at position: {self.bot.get_loc()}, outside of the map boundary")

        # update belief map
        new_observations = self.ground_truth_map.get_observation(self.bot, self.bot.get_loc())
        self.belief_map.update_map(new_observations[0], new_observations[1])

        # update exec_path
        self.bot.append_exec_loc(self.bot.get_loc())

        # count score
        occupied_locs = new_observations[0] # len of occupied cells in observation
        score = 0
        for loc in occupied_locs:
            if loc not in robot_occupied_locs:
                score += 1
        self.set_score(score)
        self.scores.append(score)

        # self._generate_data_matrices(action)

        return action

    def visualize(self, robots, curr_step, debug_occ_locs=None):
        plt.xlim(0, self.belief_map.bounds[0])
        plt.ylim(0, self.belief_map.bounds[1])
        plt.title("Step:{}".format(curr_step))

        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        
        unknown_locs = self.belief_map.get_unknown_locs()
        for spot in unknown_locs:
            hole = patches.Rectangle(spot, 1, 1, facecolor='black')
            ax.add_patch(hole)
        
        free_locs = self.belief_map.get_free_locs()
        for spot in free_locs:
            hole = patches.Rectangle(spot, 1, 1, facecolor='white')
            ax.add_patch(hole)
        
        occupied_locs = self.belief_map.get_occupied_locs()
        for spot in occupied_locs:
            hole = patches.Rectangle(spot, 1, 1, facecolor='green')
            ax.add_patch(hole)

        # used to color cells that we want to specifically see when debugging
        for spot in debug_occ_locs:
            hole = patches.Rectangle(spot, 1, 1, facecolor='yellow')
            ax.add_patch(hole)

        # plot robot
        bot_xloc = self.bot.get_loc()[0] + 0.5
        bot_yloc = self.bot.get_loc()[1] + 0.5
        plt.scatter(bot_xloc, bot_yloc, color='green', zorder=5)

        # plot robot path
        x_values = list()
        y_values = list()
        bot_exec_path = self.bot.get_exec_path()
        for loc in bot_exec_path:
            x_values.append(loc[0] + 0.5)
            y_values.append(loc[1] + 0.5)
        plt.plot(x_values, y_values)

        # plot other robot paths
        bot_comm_exec_path = set(self.bot.get_comm_exec_path())
        for bot in robots:
            if bot is not self.bot:
                other_x_values = list()
                other_y_values = list()
                other_bot_exec_path = bot.get_exec_path()
                for loc in other_bot_exec_path:
                    # plot only if path of other bot in comm_exec_path of curr bot
                    if loc in bot_comm_exec_path:
                        other_x_values.append(loc[0] + 0.5)
                        other_y_values.append(loc[1] + 0.5)
                plt.plot(other_x_values, other_y_values, zorder=1,  color='orange')

        plt.show()

    def set_score(self, score):
        self.curr_score = score

    def reset_score(self):
        self.curr_score = 0

    def get_actions(self):
        return self.actions

    def get_curr_score(self):
        return self.curr_score

    def get_scores(self):
        return self.scores

    def debug_get_net_score(self):
        return self.debug_network_score

    def debug_get_greedy_score(self):
        return self.debug_greedy_score