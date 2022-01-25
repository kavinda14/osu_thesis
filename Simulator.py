from time import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
from tqdm import tqdm
import random as random

import OraclePlanner
import NeuralNet

sys.path.insert(0, './basic_MCTS_python')
from basic_MCTS_python import mcts
# from basic_MCTS_python import plot_tree

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
    def initialize_data(self, bots_starting_loc, obs_occupied_oracle=set()):
        # self._update_map(obs_occupied_oracle)
        self.sensor_model.create_partial_info()
        # no initial observation score according to graeme
        self.sensor_model.append_score(self.score)
        self.sensor_model.append_path(self.robot.get_loc())

        self.sensor_model.create_final_path_matrix()
        # At the start, there is no action, so we just add the initial partial info into the action matrix list
        initial_partial_info_matrix = self.sensor_model.get_final_partial_info()[0]
       
        self.sensor_model.append_action_matrix(initial_partial_info_matrix)

        # this adds the starting location of the other robots into the initial path matrix
        path_matrix = self.sensor_model.get_final_path_matrices()[0]
        for loc in bots_starting_loc:
            path_matrix[loc[0]][loc[1]] = 1
        
    # train is there because of the backtracking condition in each planner 
    def run(self, neural_model, curr_robot_positions, obs_occupied_oracle=set(), train=False):
        self.iterations += 1        

        # Generate an action from the robot path
        action = OraclePlanner.random_planner(self.robot, self.sensor_model, train)
        if self.planner == "random":
            action = OraclePlanner.random_planner(self.robot, self.sensor_model, train)
        if self.planner == "greedy-o":
            action = OraclePlanner.greedy_planner(self.robot, self.sensor_model, neural_model, obs_occupied_oracle, curr_robot_positions, train, oracle=True)                                    
            # results = OraclePlanner.debug_greedy_planner(self.robot, self.sensor_model, neural_model, obs_occupied_oracle, train, self.debug_greedy_score, self.debug_network_score, oracle=True)
            # action = results[0]
            # self.debug_network_score = results[1]
            # self.debug_greedy_score = results[2]

        if self.planner == "greedy-no":
            action = OraclePlanner.greedy_planner(self.robot, self.sensor_model, neural_model, obs_occupied_oracle, curr_robot_positions, train, oracle=False)
        if self.planner == "net_everystep":
            action = OraclePlanner.greedy_planner(self.robot, self.sensor_model, neural_model, obs_occupied_oracle, curr_robot_positions, train=True, neural_net=True)
        if self.planner == "net_everyxstep":
            action = OraclePlanner.greedy_planner(self.robot, self.sensor_model, neural_model, obs_occupied_oracle, curr_robot_positions, train=True, neural_net=True)
        if self.planner == "net_nocomm":
            action = OraclePlanner.greedy_planner(self.robot, self.sensor_model, neural_model, obs_occupied_oracle, curr_robot_positions, train, neural_net=True)
        # this is to check weights created with single robot case and multi robot case
        # if self.planner == "network_wo_path":
        #     import torch
        #     import NeuralNet
        #     bounds = [21, 21]
        #     neural_model = NeuralNet.Net(bounds)
        #     neural_model.load_state_dict(torch.load("/home/kavi/thesis/neural_net_weights/circles_21x21_epoch3_random_greedyno_t800_s200_rollout")) 
        #     neural_model.eval()
        #     action = OraclePlanner.greedy_planner(self.robot, self.sensor_model, neural_model, neural_net=True)
        if self.planner == 'mcts':
            budget = 5
            max_iterations = 1000
            # max_iterations = 1
            exploration_exploitation_parameter = 50.0 # =1.0 is recommended. <1.0 more exploitation. >1.0 more exploration. 
            solution, root, list_of_all_nodes, winner_node, winner_loc = mcts.mcts(budget, max_iterations, exploration_exploitation_parameter, self.robot, self.sensor_model, self.map, self.rollout_type, self.reward_type, neural_model)
            action = self.robot.get_direction(self.robot.get_loc(), winner_loc)

        self.sensor_model.create_action_matrix(action)
        # Move the robot
        self.robot.move(action)
        # Update the explored map based on robot position
        self._update_map(obs_occupied_oracle)
        self.sensor_model.create_partial_info()
        self.sensor_model.append_score(self.score)
        self.sensor_model.append_path(self.robot.get_loc())
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