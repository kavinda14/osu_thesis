import random
import NeuralNet
import copy

def reward_random(sequence):
    return random.randint(0, 20) + len(sequence)

def reward_greedy(rollout_sequence, sensor_model, world_map, oracle=False):
    scanned_obstacles = list()
    unobs_free = copy.deepcopy(world_map.get_unobs_free())
    unobs_occupied = copy.deepcopy(world_map.get_unobs_occupied())
    reward = 0
    
    for state in rollout_sequence:
        scanned_unobs = sensor_model.scan_mcts(state.get_location(), unobs_free, unobs_occupied)
        if oracle:
            curr_scanned_obstacles = scanned_unobs[0]
        else:
            temp_curr_scanned_obstacles = scanned_unobs
            curr_scanned_obstacles = temp_curr_scanned_obstacles[0].union(temp_curr_scanned_obstacles[1])

        curr_reward = 0
        # this makes sure that reward is calculated for total UNIQUE obstacles that are scanned
        for loc in curr_scanned_obstacles:
            if loc not in scanned_obstacles:
                scanned_obstacles.append(loc)
                curr_reward += 1
        reward += curr_reward

    return reward

def reward_network(rollout_sequence, sensor_model, world_map, neural_model):
    # the map should be updating as we are iterating through the sequence
    # if not, it is taking the old map and just doing that
    # pass the action_loc to the action matrix function instead of the actual action
    reward_final_path = copy.copy(sensor_model.get_final_path()) # these are the executed paths + all the incremental rollout paths
    # we don't need other_paths here because it is already handled by the create_final_path_matrix_mcts() function
    reward_map = copy.deepcopy(world_map)
    
    partial_info = [sensor_model.create_partial_info_mcts(reward_map, False)]
    partial_info_binary_matrices = sensor_model.create_binary_matrices(partial_info)

    reward = 0
    for state in rollout_sequence:
        loc = state.get_location()
        # if tuple(loc) in reward_final_path or tuple(loc) in reward_final_other_path:
        #     continue
        reward_final_path.append(state.get_location())

        path_matrix = sensor_model.create_final_path_matrix_mcts(reward_final_path, update=False)
        
        final_actions = [sensor_model.create_action_matrix_mcts(loc)]
        final_actions_binary_matrices = sensor_model.create_binary_matrices(final_actions)

        input = NeuralNet.create_image(partial_info_binary_matrices, path_matrix, final_actions_binary_matrices)
        input = input.unsqueeze(0).float()
        
        reward += neural_model(input).item()

    return reward

