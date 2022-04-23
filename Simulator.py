import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

sys.path.insert(0, './basic_MCTS_python')
from basic_MCTS_python import mcts
from basic_MCTS_python import plot_tree

class Simulator:
    def __init__(self, belief_map, ground_truth_map, bot, sensor_model, generate_data):
        self.ground_truth_map = ground_truth_map
        self.belief_map = belief_map
        self.bot = bot
        self.sensor_model = sensor_model

        self.curr_score = 0
        self.final_scores = list()

        # Simulator.py is used for eval and data generation
        self.generate_data = generate_data

        self.debug_network_score = list()
        self.debug_greedy_score = list()

    # creates the initially matrices needed
    def initialize_data(self, robot_start_locs):
        self.sensor_model.create_partial_info()
        self.final_scores.append(self.curr_score)  # init score is 0

        # at the start, there is no action, so we just add the initial partial info into the action matrix list
        partial_info_matrix = self.sensor_model.get_partial_info_matrices()[0]
        self.sensor_model.append_action_matrix(partial_info_matrix)

        # to initialize a matrix in comm_path_matrices for data generation ONLY
        if self.generate_data:
            curr_bot_loc = self.bot.get_loc()
            # keep in mind that for rollout in data generation, we create the path matrices separately and then combine them
            self.sensor_model.create_rollout_path_matrix()
            self.sensor_model.create_rollout_comm_path_matrix()

            path_matrix = self.sensor_model.get_path_matrices()[0]
            path_matrix[curr_bot_loc[0]][curr_bot_loc[1]] = 1

            path_matrix = self.sensor_model.get_comm_path_matrices()[0]
            for loc in robot_start_locs:
                if loc != curr_bot_loc:
                    path_matrix[loc[0]][loc[1]] = 1
        else:
            # this contains paths and other paths
            self.sensor_model.create_path_matrix()

            # this adds the starting location of the other robots into the initial path matrix
            path_matrix = self.sensor_model.get_path_matrices()[0]
            for loc in robot_start_locs:
                path_matrix[loc[0]][loc[1]] = 1
        

    # train is there because of the backtracking condition in each planner 
    def run(self, planner, robot_curr_locs, robot_occupied_locs, neural_model, device=None, generate_data=False):       
        action = planner.get_action(self.bot, robot_curr_locs)

        self.sensor_model.create_action_matrix(action, self.bot.get_loc())

        self.bot.move(action)

        # sanity check the robot is in bounds after moving
        if not self.ground_truth_map.is_valid_loc(self.bot.get_loc()):
            raise ValueError(
                f"Robot has left the map. It is at position: {self.bot.get_loc()}, outside of the map boundary")

        # update belief map
        new_observations = self.ground_truth_map.get_observation(self.bot, self.bot.get_loc())
        occupied_locs = new_observations[0] # len of occupied cells in observation
        
        score = 0
        for loc in occupied_locs:
            if loc not in robot_occupied_locs:
                score += 1
        self.set_score(score)
        
        self.belief_map.update_map(new_observations[0], new_observations[1])

        # create matrices/lists for net
        self.final_scores.append(score)
        self.sensor_model.create_partial_info()
        self.bot.append_exec_paths(self.bot.get_loc())
        if generate_data:
            self.sensor_model.create_final_rollout_path_matrix()
            self.sensor_model.create_final_rollout_other_path_matrix()
        else:
            self.sensor_model.create_path_matrix()

    def visualize(self, robots, step):
        plt.xlim(0, self.belief_map.bounds[0])
        plt.ylim(0, self.belief_map.bounds[1])
        plt.title("Planner: {}, Score: {} Step:{}".format(self.planner, sum(self.sensor_model.get_final_scores()), step))

        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        
        for spot in self.belief_map.unobs_occupied:
            hole = patches.Rectangle(spot, 1, 1, facecolor='red')
            ax.add_patch(hole)

        for spot in self.belief_map.unobs_free:
            hole = patches.Rectangle(spot, 1, 1, facecolor='black')
            ax.add_patch(hole)
        
        for spot in self.belief_map.obs_free:
            hole = patches.Rectangle(spot, 1, 1, facecolor='white')
            ax.add_patch(hole)
        
        for spot in self.belief_map.obs_occupied:
            hole = patches.Rectangle(spot, 1, 1, facecolor='green')
            ax.add_patch(hole)

        # Plot robot
        robot_x = self.bot.get_loc()[0] + 0.5
        robot_y = self.bot.get_loc()[1] + 0.5
        plt.scatter(robot_x, robot_y, color='purple', zorder=5)

        # Plot robot path
        x_values = list()
        y_values = list()
        for path in self.sensor_model.get_final_path():
            x_values.append(path[0] + 0.5)
            y_values.append(path[1] + 0.5)
        plt.plot(x_values, y_values)

        # Plot other robot paths
        for bot in robots:
            sensor_model = bot.get_sensor_model()
            if bot is not self.bot:
                x_values_other = list()
                y_values_other = list()
                bot_path = sensor_model.get_final_path()
                for path in bot_path:
                    if path in set(self.sensor_model.get_final_other_path()):
                        x_values_other.append(path[0] + 0.5)
                        y_values_other.append(path[1] + 0.5)
                plt.plot(x_values_other, y_values_other, zorder=1,  color='orange')

        plt.show()

    def get_score(self):
        return self.curr_score

    def set_score(self, score):
        self.curr_score = score

    def reset_score(self):
        self.curr_score = 0

    def get_actions(self):
        return self.actions

    def debug_get_net_score(self):
        return self.debug_network_score

    def debug_get_greedy_score(self):
        return self.debug_greedy_score
