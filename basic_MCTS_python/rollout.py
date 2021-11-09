'''
Basic MCTS implementation
Graeme Best
Oregon State University
Jan 2020
'''

from cost import cost
import random
import copy
import NeuralNet
import torch

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

    actions = ['left', 'right', 'forward', 'backward']
    for action in actions:
        valid, new_loc = robot.check_valid_move_mcts(action, current_loc, True)
        if valid:
        # if valid and new_loc not in sequence:
            # this makes the rollout not backtrack (might be too strict)
            # sequence.append(new_loc)
            neighbors.append(State(action, new_loc))

    return neighbors

def rollout_random(subsequence, budget, robot):
    # THESE ARE STATES
    current_state = subsequence[-1]
    sequence = copy.deepcopy(subsequence)
    while cost(sequence) < budget:
        neighbors = generate_valid_neighbors(current_state, subsequence, robot)
        r = random.randint(0, len(neighbors)-1)
        next_state = neighbors[r]
        sequence.append(next_state)
    
    return sequence

def rollout_greedy(subsequence, budget, robot, sensor_model, oracle=False):
    # rollout_final_path = copy.deepcopy(sensor_model.get_final_path())
    sequence = copy.deepcopy(subsequence)

    # THESE ARE STATES
    current_state = subsequence[-1]
   
    while cost(sequence) < budget:
        neighbors = generate_valid_neighbors(current_state, subsequence, robot)
        best_state = None
        # use -infinity because network outputs negative values sometimes
        best_action_score = float("-inf")

        for state in neighbors:
            scanned_unobs = sensor_model.scan(state.get_location(), False)
            if oracle: 
                action_score = len(scanned_unobs[0])
            else:  
                action_score = len(scanned_unobs[0]) + len(scanned_unobs[1])
            
            if action_score > best_action_score:
                best_action_score = action_score
                best_state = state
        
        sequence.append(best_state)
        # rollout_final_path.append(best_state.get_location())
        # this is where the robot "moves"
        current_state = best_state
        best_action_score = float("-inf")
        best_state = None

    return sequence

def rollout_network(subsequence, budget, robot, sensor_model, world_map, neural_model):
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
            action_score = neural_model(input).item()
            
            if action_score > best_action_score:
                best_action_score = action_score
                best_state = state
        
        sequence.append(best_state)
        rollout_final_path.append(best_state.get_location())
        # this is where the robot "moves"
        current_state = best_state
        best_action_score = float("-inf")
        best_state = None

    return sequence





