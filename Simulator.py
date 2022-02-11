from time import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
from torch import randint
from tqdm import tqdm
import random as random

import OraclePlanner
import NeuralNet
import torch

sys.path.insert(0, './basic_MCTS_python')
from basic_MCTS_python import mcts
from basic_MCTS_python import plot_tree

class Simulator:
    def __init__(self, world_map, robot, sensor_model, planner, rollout_type="random", reward_type="random"):
        """
        Inputs:
        world_map: The map to be explored
        robot: the Robot object from Robot.py
        """
        self.map = world_map
        self.robot = robot
        self.sensor_model = sensor_model
        # The following identify what has been seen by the robot
        self.obs_occupied = set()
        self.obs_free = set()

        self.score = 0
        self.iterations = 0

        self.planner = planner
        self.rollout_type = rollout_type
        self.reward_type = reward_type

        self.debug_network_score = list()
        self.debug_greedy_score = list()
    
    def debug_get_net_score(self):
        return self.debug_network_score

    def debug_get_greedy_score(self):
        return self.debug_greedy_score

    # new addition to multi-robot code
    def initialize_data(self, bots_starting_loc, obs_occupied_oracle=set(), generate_data=False):
        # self._update_map(obs_occupied_oracle)
        self.sensor_model.create_partial_info()
        # no initial observation score according to graeme
        self.sensor_model.append_score(self.score)
        curr_bot_loc = self.robot.get_loc()
        self.sensor_model.append_path(curr_bot_loc)

        # At the start, there is no action, so we just add the initial partial info into the action matrix list
        initial_partial_info_matrix = self.sensor_model.get_final_partial_info()[0]
       
        self.sensor_model.append_action_matrix(initial_partial_info_matrix)

        # to initialize a matrix in final_other_path_matrices for data generation ONLY
        if generate_data:
            self.sensor_model.create_final_rollout_path_matrix()
            self.sensor_model.create_final_rollout_other_path_matrix()

            path_matrix = self.sensor_model.get_final_path_matrices()[0]
            path_matrix[curr_bot_loc[0]][curr_bot_loc[1]] = 1

            path_matrix = self.sensor_model.get_final_other_path_matrices()[0]
            for loc in bots_starting_loc:
                if loc != curr_bot_loc:
                    path_matrix[loc[0]][loc[1]] = 1
        else:
            # this contains paths and other paths
            self.sensor_model.create_final_path_matrix()

            # this adds the starting location of the other robots into the initial path matrix
            path_matrix = self.sensor_model.get_final_path_matrices()[0]
            for loc in bots_starting_loc:
                path_matrix[loc[0]][loc[1]] = 1
        

    # train is there because of the backtracking condition in each planner 
    def run(self, neural_model, curr_robot_positions, neural_model_trial=None, device=None, obs_occupied_oracle=set(), train=False, generate_data=False, debug_mcts_reward_greedy_list=list(), debug_mcts_reward_network_list=list(), CONF=None, json_comp_conf=None):
        self.iterations += 1        

        # Generate an action from the robot path
        action = OraclePlanner.random_planner(self.robot, self.sensor_model, train)
        if self.planner == "random":
            action = OraclePlanner.random_planner(self.robot, self.sensor_model, train)
        if self.planner in ("greedy-o", "greedy-o_everyxstep"):
            action = OraclePlanner.greedy_planner(self.robot, self.sensor_model, neural_model, obs_occupied_oracle, curr_robot_positions, train, oracle=True)                                    
        if self.planner in ("greedy-no", "greedy-no_everyxstep"):
            action = OraclePlanner.greedy_planner(self.robot, self.sensor_model, neural_model, obs_occupied_oracle, curr_robot_positions, train, oracle=False)
        if self.planner in ("net_everystep", "net_everyxstep", "net_nocomm"):
            action = OraclePlanner.greedy_planner(self.robot, self.sensor_model, neural_model, obs_occupied_oracle, curr_robot_positions, train=True, neural_net=True, device=device)
        if self.planner == "net_trial":
            action = OraclePlanner.greedy_planner(self.robot, self.sensor_model, neural_model_trial, obs_occupied_oracle, curr_robot_positions, train=True, neural_net=True, device=device)
        if self.planner == 'mcts':
            budget = 6
            max_iterations = 1000
            # max_iterations = 5
            # exploration_exploitation_parameter = 10.0 # =1.0 is recommended. <1.0 more exploitation. >1.0 more exploration. 
            exploration_exploitation_parameter = 5.0 # =1.0 is recommended. <1.0 more exploitation. >1.0 more exploration. 
            solution, solution_locs, root, list_of_all_nodes, winner_node, winner_loc = mcts.mcts(budget, max_iterations, exploration_exploitation_parameter, self.robot, self.sensor_model, self.map, self.rollout_type, self.reward_type, neural_model, debug_mcts_reward_greedy_list, 
                                                                                                  debug_mcts_reward_network_list, device=device, CONF=CONF, json_comp_conf=json_comp_conf)
            # plot_tree.plotTree(list_of_all_nodes, winner_node, False, budget, "1", exploration_exploitation_parameter)
            
            # times_visited = self.sensor_model.get_final_path().count(tuple(winner_loc)) + self.sensor_model.get_final_other_path().count(tuple(winner_loc))
            
            # 2 robots cannot be in the same loc condition
            # times_visited is for backtracking
            # if tuple(winner_loc) in curr_robot_positions or times_visited > 1:
            if tuple(winner_loc) in curr_robot_positions:
                idx = random.randint(0, len(solution_locs)-1)
                winner_loc = solution_locs[idx]

            curr_robot_positions.add(tuple(winner_loc))
            action = self.robot.get_direction(self.robot.get_loc(), winner_loc)

        self.sensor_model.create_action_matrix(action)
        # Move the robot
        self.robot.move(action)
        # Update the explored map based on robot position
        self._update_map(obs_occupied_oracle)
        self.sensor_model.create_partial_info()
        self.sensor_model.append_score(self.score)
        self.sensor_model.append_path(self.robot.get_loc())
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
        self.score = 0
        self.obs_occupied = set()
        self.obs_free = set()

    def get_score(self):
        return self.score

    def set_score(self, score):
        self.score = score

    def reset_score(self):
        self.score = 0

    def get_iteration(self):
        return self.iterations

    def get_actions(self):
        return self.actions

    def get_obs_free(self):
        return self.obs_free
    
    def get_obs_occupied(self):
        return self.obs_occupied

    def _update_map(self, obs_occupied_oracle):
        # Sanity check the robot is in bounds
        if not self.robot.check_valid_loc():
            print(self.robot.get_loc())
            raise ValueError(f"Robot has left the map. It is at position: {self.robot.get_loc()}, outside of the map boundary")
        
        new_observations = self.sensor_model.scan(self.robot.get_loc(), obs_occupied_oracle)
        # Score is the number of new obstacles found
        # random thing is just for debugging - delete after testing
        self.set_score(len(new_observations[0]))
        self.obs_occupied = self.obs_occupied.union(new_observations[0])
        self.obs_free = self.obs_free.union(new_observations[1])
        self.map.obs_occupied = self.map.obs_occupied.union(self.obs_occupied)
        self.map.obs_free = self.map.obs_free.union(self.obs_free)

    def visualize(self):
        plt.xlim(0, self.map.bounds[0])
        plt.ylim(0, self.map.bounds[1])
        plt.title("Planner: {}, Score: {}".format(self.planner, sum(self.sensor_model.get_final_scores())))

        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        
        for spot in self.map.unobs_occupied:
            hole = patches.Rectangle(spot, 1, 1, facecolor='red')
            ax.add_patch(hole)

        for spot in self.map.unobs_free:
            hole = patches.Rectangle(spot, 1, 1, facecolor='black')
            ax.add_patch(hole)
        
        for spot in self.obs_free:
            hole = patches.Rectangle(spot, 1, 1, facecolor='white')
            ax.add_patch(hole)
        
        for spot in self.obs_occupied:
            hole = patches.Rectangle(spot, 1, 1, facecolor='green')
            ax.add_patch(hole)

        # Plot robot
        robot_x = self.robot.get_loc()[0] + 0.5
        robot_y = self.robot.get_loc()[1] + 0.5
        plt.scatter(robot_x, robot_y, color='purple', zorder=5)

        # Plot robot path
        x_values = list()
        y_values = list()
        for path in self.sensor_model.get_final_path():
            x_values.append(path[0] + 0.5)
            y_values.append(path[1] + 0.5)

        plt.plot(x_values, y_values)


        plt.show()