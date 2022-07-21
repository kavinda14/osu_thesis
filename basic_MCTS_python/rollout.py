from cost import cost
import random
import copy
import NeuralNet
from State import generate_valid_neighbors


def rollout_random(subsequence, budget, bot):
    curr_state = subsequence[-1]
    sequence = copy.copy(subsequence)

    bot_belief_map = bot.get_belief_map()
    while cost(sequence) < budget:
        neighbors = generate_valid_neighbors(curr_state, bot_belief_map)
        r = random.randint(0, len(neighbors)-1)
        next_state = neighbors[r]
        sequence.append(next_state)
        curr_state = next_state

    return sequence

def rollout_cellcount(subsequence, budget, bot):
    sequence = copy.copy(subsequence)

    bot_belief_map = bot.get_belief_map()
    bot_sense_range = bot.get_sense_range()
   
    # these are State objects
    curr_state = subsequence[-1]
   
    while cost(sequence) < budget:
        neighbors = generate_valid_neighbors(curr_state, bot_belief_map)
        best_state = None
        # use -infinity because network outputs negative values sometimes
        best_action_score = float("-inf")

        for state in neighbors:
            potential_loc = tuple(state.get_loc())
               
            action_score = len(bot_belief_map.count_unknown_cells(bot_sense_range, potential_loc))

            if action_score > best_action_score:
                best_action_score = action_score
                best_state = state
        
        sequence.append(best_state)
        # this is where the robot "moves"
        curr_state = best_state
        best_action_score = float("-inf")
        best_state = None

    return sequence

def rollout_network(subsequence, budget, bot, neural_model, device):
    bot_sensor_model = bot.get_sensor_model()
    bot_belief_map = bot.get_belief_map()

    sequence = copy.copy(subsequence)
    # paths already traversed before mcts     
    exec_path = bot.get_exec_path()
    rollout_path = copy.copy(exec_path)

    # partial_info = [bot_sensor_model.create_partial_info(False)]
    # partial_info_binary_matrices = bot_sensor_model.create_binary_matrices(partial_info)

    # these are State objects
    curr_state = subsequence[-1]
   
    while cost(sequence) < budget:
        curr_bot_loc = curr_state.get_loc()
        neighbors = generate_valid_neighbors(curr_state, bot_belief_map)

        # path_matrix = bot_sensor_model.create_path_matrix(False, rollout_path)
        
        # we can't initialize this to None because we are skipping exec_path in the for loop
        r = random.randint(0, len(neighbors)-1)
        best_state = neighbors[r]
        # use -infinity because network outputs negative values sometimes
        best_action_score = float("-inf")

        for state in neighbors:
            action = state.get_action()
            action_matrix = [bot_sensor_model.create_action_matrix(action, curr_bot_loc, True)]
            action_binary_matrices = bot_sensor_model.create_binary_matrices(action_matrix)
            path_matrix = bot_sensor_model.create_path_matrix(action, curr_bot_loc, get_matrix=True, input_path=rollout_path)
           
            input = NeuralNet.create_image(path_matrix, action_binary_matrices)
            # The unsqueeze adds an extra dimension at index 0 and the .float() is needed otherwise PyTorch will complain
            # By unsqeezing, we add a batch dimension to the input, which is required by PyTorch: (n_samples, channels, height, width) 
            input = input.unsqueeze(0).float().to(device)
            action_score = neural_model(input).item()
            
            if action_score > best_action_score:
                best_action_score = action_score
                best_state = state
        
        sequence.append(best_state)
        rollout_path.append(best_state.get_loc())
        # this is where the robot "moves"
        curr_state = best_state
        best_action_score = float("-inf")
        r = random.randint(0, len(neighbors)-1)
        best_state = neighbors[r]

    return sequence





