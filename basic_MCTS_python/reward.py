'''
Basic MCTS implementation
Graeme Best
Oregon State University
Jan 2020
'''

from action import Action #, printActionSequence
import random
import SensorModel
import NeuralNet
import torch
import copy


def reward_random(sequence):
    return random.randint(0, 10) + len(sequence)

def reward_greedy(rollout_sequence, sensor_model, oracle=False):
    reward = 0
    scanned_obstacles = list()
    
    for state in rollout_sequence:
        if oracle:
            curr_scanned_obstacles = sensor_model.scan(state.get_location(), False)[0]
        else:
            temp_curr_scanned_obstacles = sensor_model.scan(state.get_location(), False)
            curr_scanned_obstacles = temp_curr_scanned_obstacles[0].union(temp_curr_scanned_obstacles[1])

        curr_reward = 0
        for loc in curr_scanned_obstacles:
            if loc not in scanned_obstacles:
                scanned_obstacles.append(loc)
                curr_reward += 1
        
        reward += curr_reward

    return reward

def reward_network(rollout_sequence, sensor_model, world_map):
    model = NeuralNet.Net(world_map.get_bounds())
    model.load_state_dict(torch.load("/home/kavi/thesis/neural_net_weights/circles_random_21x21"))
    model.eval()

    reward_final_path = copy.deepcopy(sensor_model.get_final_path())
    reward_map = copy.deepcopy(world_map)

    partial_info = [sensor_model.create_partial_info_mcts(reward_map, False)]
    partial_info_binary_matrices = sensor_model.create_binary_matrices(partial_info)

    reward = 0
    for state in rollout_sequence:
        action = state.get_action()
        # fix this bug where root is being added to rollout sequence
        if action == "root":
            break
        reward_final_path.append(state.get_location())
        path_matrix = sensor_model.create_final_path_matrix_mcts(reward_final_path, False)
        final_actions = [sensor_model.create_action_matrix(action, True)]
        final_actions_binary_matrices = sensor_model.create_binary_matrices(final_actions)
        input = NeuralNet.create_image(partial_info_binary_matrices, path_matrix, final_actions_binary_matrices)
        input = input.unsqueeze(0).float()
        
        reward += model(input).item()

    return reward

