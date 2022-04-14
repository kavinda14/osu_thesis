from time import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import random as random

import Planners

sys.path.insert(0, './basic_MCTS_python')
from basic_MCTS_python import mcts
from basic_MCTS_python import plot_tree

class Simulator:
    def __init__(self, belief_map, bot, sensor_model, planner, generate_data):
        self.belief_map = belief_map
        self.bot = bot
        self.sensor_model = sensor_model

        self.curr_score = 0
        self.final_scores = list()
        self.iterations = 0

        self.planner = planner

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
    def run(self, neural_model, curr_robot_positions, neural_model_trial=None, device=None, obs_occupied_oracle=set(), train=False, generate_data=False, debug_mcts_reward_greedy_list=list(), debug_mcts_reward_network_list=list(), CONF=None, json_comp_conf=None):
        self.iterations += 1        

        # Generate an action from the robot path
        action = None
        if self.planner in ("random_fullcomm", "random_partialcomm", "random_poorcomm"):
            action = Planners.random_planner(self.bot, curr_robot_positions)
        elif self.planner in ("greedy_fullcomm", "greedy_partialcomm", "greedy_poorcomm"):
            action = Planners.greedy_planner(self.bot, self.sensor_model, neural_model, curr_robot_positions)
        elif self.planner in ("net_fullcomm", "net_partialcomm", "net_poorcomm"):
            action = Planners.greedy_planner(self.bot, self.sensor_model, neural_model, curr_robot_positions, neural_net=True, device=device)
        elif self.planner == "net_trial":
            action = Planners.greedy_planner(self.bot, self.sensor_model, neural_model_trial, curr_robot_positions, neural_net=True, device=device)
        elif self.planner == 'mcts':
            budget = 6
            max_iterations = 1000
            exploration_exploitation_parameter = 10.0 # =1.0 is recommended. <1.0 more exploitation. >1.0 more exploration.
            solution, solution_locs, root, list_of_all_nodes, winner_node, winner_loc = mcts.mcts(budget, max_iterations, exploration_exploitation_parameter, self.bot, self.sensor_model, self.belief_map, self.rollout_type, self.reward_type, neural_model, debug_mcts_reward_greedy_list, 
                                                                                                  debug_mcts_reward_network_list, device=device, CONF=CONF, json_comp_conf=json_comp_conf)
            # plot_tree.plotTree(list_of_all_nodes, winner_node, False, budget, "1", exploration_exploitation_parameter)
            
            times_visited = self.sensor_model.get_final_path().count(tuple(winner_loc)) + self.sensor_model.get_final_other_path().count(tuple(winner_loc))
            # 2 robots cannot be in the same loc condition
            # times_visited is for backtracking
            if (tuple(winner_loc) in curr_robot_positions) or (times_visited >= 4):
                idx = random.randint(0, len(solution_locs)-1)
                winner_loc = solution_locs[idx]

            curr_robot_positions.add(tuple(winner_loc))
            action = self.bot.get_direction(self.bot.get_loc(), winner_loc)
        else:
            print("Wrong planner name!")

        self.sensor_model.create_action_matrix(action, self.bot.get_loc())
        # Move the robot
        self.bot.move(action)
        # Update the explored map based on robot position
        self._update_map(obs_occupied_oracle)
        self.sensor_model.create_partial_info()
        self.sensor_model.append_score(self.curr_score)
        self.sensor_model.append_path(self.bot.get_loc())
        if generate_data:
            self.sensor_model.create_final_rollout_path_matrix()
            self.sensor_model.create_final_rollout_other_path_matrix()
        else:
            self.sensor_model.create_final_path_matrix()

        # Score is calculated in _update function.
        # It needs to be reset otherwise the score will carry on to the next iteration even if no new obstacles were scanned.
        self.reset_score()

        # End when all objects have been observed
        # if (len(self.obs_occupied) == self.map.unobs_occupied):
        #     return True
        # else:
        #     return False

    def reset_game(self):
        self.iterations = 0
        self.curr_score = 0
        self.obs_occupied = set()
        self.obs_free = set()

    def _update_map(self, obs_occupied_oracle):
        # Sanity check the robot is in bounds
        if not self.bot.check_valid_loc():
            print(self.bot.get_loc())
            raise ValueError(f"Robot has left the map. It is at position: {self.bot.get_loc()}, outside of the map boundary")
        
        new_observations = self.sensor_model.scan(self.bot.get_loc(), obs_occupied_oracle)
        # Score is the number of new obstacles found
        # random thing is just for debugging - delete after testing
        self.set_score(len(new_observations[0]))
        self.belief_map.obs_occupied = self.belief_map.obs_occupied.union(new_observations[0])
        self.belief_map.obs_free = self.belief_map.obs_free.union(new_observations[1])

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

    def get_iteration(self):
        return self.iterations

    def get_actions(self):
        return self.actions

    def debug_get_net_score(self):
        return self.debug_network_score

    def debug_get_greedy_score(self):
        return self.debug_greedy_score
