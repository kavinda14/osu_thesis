import random
import NeuralNet
import copy

def reward_random(sequence):
    return random.randint(0, 20) + len(sequence)

def reward_cellcount(rollout_sequence, bot):
    bot_belief_map = bot.get_belief_map()
    bot_sense_range = bot.get_sense_range()
    
    unknown_cells = set()
    for state in rollout_sequence:
        potential_loc = state.get_loc() 
        curr_unknown_cells = bot_belief_map.count_unknown_cells(bot_sense_range, potential_loc)
        unknown_cells = unknown_cells.union(curr_unknown_cells)
    
    # reward is the unique unknown locs
    return len(unknown_cells)

def reward_network(rollout_sequence, bot, neural_model, device):
    bot_sensor_model = bot.get_sensor_model()

    exec_path_copy = copy.copy(bot.get_exec_path()) # these are the executed paths + all the incremental rollout paths
    # we don't need other_paths here because it is already handled by create_path_matrix()
    
    partial_info = [bot_sensor_model.create_partial_info(False)]
    partial_info_binary_matrices = bot_sensor_model.create_binary_matrices(partial_info)

    curr_bot_loc = bot.get_loc()
    reward = 0
    for state in rollout_sequence:
        path_matrix = bot_sensor_model.create_path_matrix(False, exec_path_copy)
        
        action = state.get_action()
        action_matrix = [bot_sensor_model.create_action_matrix(action, curr_bot_loc, True)]
        action_binary_matrices = bot_sensor_model.create_binary_matrices(action_matrix)

        input = NeuralNet.create_image(partial_info_binary_matrices, path_matrix, action_binary_matrices)
        input = input.unsqueeze(0).float().to(device)
        
        reward += neural_model(input).item()

        curr_bot_loc = state.get_loc()
        exec_path_copy.append(curr_bot_loc) 

    return reward

