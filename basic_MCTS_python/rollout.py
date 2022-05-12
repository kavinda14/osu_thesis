from torch import randint
from cost import cost
import random
import copy
import NeuralNet
from utils import State, generate_valid_neighbors

def rollout_random(subsequence, budget, robot):
    # THESE ARE STATES
    current_state = subsequence[-1]
    # sequence = copy.deepcopy(subsequence)
    sequence = copy.copy(subsequence)

    while cost(sequence) < budget:
        neighbors = generate_valid_neighbors(current_state, subsequence, robot)
        r = random.randint(0, len(neighbors)-1)
        next_state = neighbors[r]
        sequence.append(next_state)
    
    return sequence

def rollout_cellcount(subsequence, budget, robot, sensor_model, world_map, oracle=False):
    sequence = copy.copy(subsequence)
    unobs_free = world_map.get_unobs_free()
    unobs_occupied = world_map.get_unobs_occupied()
    executed_paths = sensor_model.get_final_path()
    other_executed_paths = sensor_model.get_final_other_path()
   
    # these are State objects
    current_state = subsequence[-1]
   
    while cost(sequence) < budget:
        neighbors = generate_valid_neighbors(current_state, subsequence, robot)
        best_state = None
        # use -infinity because network outputs negative values sometimes
        best_action_score = float("-inf")

        # count was added because of other_executed_paths..
        # if not what happens is the entire for loop is skipped because non of the States() in subsequence..
        # ..gets iterated over because they have all been traversed 
        count = 0
        for state in neighbors:
            count += 1
            if count != len(neighbors):
                loc = tuple(state.get_location())
                # times_visited = executed_paths.count(loc) + other_executed_paths.count(loc)
                if loc in (executed_paths, other_executed_paths):
                # if times_visited >= 1:
                    continue
            else: 
                index = random.randint(0, len(neighbors)-1)
                state = neighbors[index]
                scanned_unobs = sensor_model.scan_mcts(state.get_location(), unobs_free, unobs_occupied)
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

def rollout_network(subsequence, budget, robot, sensor_model, world_map, neural_model, device):
    rollout_final_path = copy.copy(sensor_model.get_final_path())

    sequence = copy.copy(subsequence)
    # paths already traversed before mcts     
    executed_paths = sensor_model.get_final_path()
    other_executed_paths = sensor_model.get_final_other_path()

    partial_info = [sensor_model.create_partial_info_mcts(unobs_free=world_map.get_unobs_free(),
    unobs_occupied=world_map.get_unobs_occupied(), obs_occupied=world_map.get_obs_occupied(),
    obs_free=world_map.get_obs_free(), bounds=world_map.get_bounds(), update=False)]

    partial_info_binary_matrices = sensor_model.create_binary_matrices(partial_info)

    # these are State objects
    current_state = subsequence[-1]
   
    while cost(sequence) < budget:
        path_matrix = sensor_model.create_final_path_matrix_mcts(rollout_final_path, False)
        neighbors = generate_valid_neighbors(current_state, subsequence, robot)
        # we can't initialize this to None because we are skipping executed_paths in the for loop
        best_state = neighbors[0]
        # use -infinity because network outputs negative values sometimes
        best_action_score = float("-inf")

        for state in neighbors:
            loc = tuple(state.get_location())
            # times_visited = executed_paths.count(loc) + other_executed_paths.count(loc)
            if loc in (executed_paths, other_executed_paths):
            # if times_visited >= 1:
                continue
            action = state.get_action()
            final_actions = [sensor_model.create_action_matrix(action, True)]
            final_actions_binary_matrices = sensor_model.create_binary_matrices(final_actions)
            input = NeuralNet.create_image(partial_info_binary_matrices, path_matrix, final_actions_binary_matrices)
            # The unsqueeze adds an extra dimension at index 0 and the .float() is needed otherwise PyTorch will complain
            # By unsqeezing, we add a batch dimension to the input, which is required by PyTorch: (n_samples, channels, height, width) 
            input = input.unsqueeze(0).float().to(device)
            action_score = neural_model(input).item()
            
            if action_score > best_action_score:
                best_action_score = action_score
                best_state = state
        
        sequence.append(best_state)
        rollout_final_path.append(best_state.get_location())
        # this is where the robot "moves"
        current_state = best_state
        best_action_score = float("-inf")
        best_state = neighbors[0]

    return sequence





