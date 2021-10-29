'''
Basic MCTS implementation
Graeme Best
Oregon State University
Jan 2020
'''

from numpy.core.numeric import Infinity
from cost import cost
import random
import copy
import NeuralNet
import torch
# from mcts import State

# def rollout(subsequence, action_set, budget):
#     # Random rollout policy
#     # Pick random actions until budget is exhausted
#     num_actions = len(action_set)
#     if num_actions <= 0:
#         raise ValueError('rollout: num_actions is ' + str(num_actions))
#     sequence = copy.deepcopy(subsequence)
#     while cost(sequence) < budget:
#         r = random.randint(0,num_actions-1)
#         sequence.append(action_set[r])

#     return sequence

class State():
    def __init__(self, action, location):
        self.action = action
        self.location = location

    def get_action(self):
        return self.action
    
    def get_location(self):
        return self.location

def generate_valid_neighbors(current_state, state_sequence, robot):
    neighbors = list()
    current_loc = current_state.get_location()

    sequence = [state.get_location() for state in state_sequence]
    actions = ['left', 'right', 'forward', 'backward']
    for action in actions:
        valid, new_loc = robot.check_valid_move_mcts(action, current_loc, True)
        if valid:
        # if valid and new_loc not in sequence:
            # this makes the rollout not backtrack (might be too strict)
            # sequence.append(new_loc)
            neighbors.append(State(action, new_loc))

    return neighbors

def rollout(subsequence, budget, robot):
    # THESE ARE STATES
    current_state = subsequence[-1]
    current_loc = subsequence[-1].get_location()
    # print('rollout current_loc', current_loc)
    sequence = copy.deepcopy(subsequence)
    while cost(sequence) < budget:
        neighbors = generate_valid_neighbors(current_state, subsequence, robot)
        # for neighbor in neighbors:
        #     print('rollout neigh coords: ', neighbor.get_location())
        # print(len(neighbors))
        r = random.randint(0, len(neighbors)-1)
        next_state = neighbors[r]
        sequence.append(next_state)
        current_loc = next_state.get_location()
    
    return sequence

def rollout_network(subsequence, budget, robot, sensor_model, world_map):
    model = NeuralNet.Net(world_map.get_bounds())
    model.load_state_dict(torch.load("/home/kavi/thesis/neural_net_weights/circles_random_21x21"))
    model.eval()

    rollout_final_path = copy.deepcopy(sensor_model.get_final_path())
    rollout_map = copy.deepcopy(world_map)
    sequence = copy.deepcopy(subsequence)

    partial_info = [sensor_model.create_partial_info_mcts(rollout_map, False)]
    partial_info_binary_matrices = sensor_model.create_binary_matrices(partial_info)

    # THESE ARE STATES
    current_state = subsequence[-1]
   
    while cost(sequence) < budget:
        path_matrix = sensor_model.create_final_path_matrix_mcts(rollout_final_path, False)
        neighbors = generate_valid_neighbors(current_state, subsequence, robot)
        best_state = None
        # use -infinity because network outputs negative values sometimes
        best_action_score = float("-inf")

        for state in neighbors:
            action = state.get_action()
            final_actions = [sensor_model.create_action_matrix(action, True)]
            final_actions_binary_matrices = sensor_model.create_binary_matrices(final_actions)
            input = NeuralNet.create_image(partial_info_binary_matrices, path_matrix, final_actions_binary_matrices)
            # The unsqueeze adds an extra dimension at index 0 and the .float() is needed otherwise PyTorch will complain
            # By unsqeezing, we add a batch dimension to the input, which is required by PyTorch: (n_samples, channels, height, width) 
            input = input.unsqueeze(0).float()
            action_score = model(input).item()
            
            if action_score > best_action_score:
                best_action_score = action_score
                best_state = state
        
        sequence.append(best_state)
        rollout_final_path.append(best_state.get_location())
        best_action_score = float("-inf")
        best_state = None

    return sequence





