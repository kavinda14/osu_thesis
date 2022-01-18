import random
import torch
import NeuralNet

def random_planner(robot, sensor_model, train):
    actions = ['left', 'right', 'backward', 'forward']
    valid_move = False # Checks if the pixel is free
    visited_before = True # Check if the pixel has been visited before
    action = random.choice(actions)
    action = random.choice(actions)
   
    counter = 0
    while True:
        counter += 1
        action = random.choice(actions)
        valid_move = robot.check_valid_move(action) 
        potential_next_loc = robot.get_action_loc(action)

        # only in data generation do we want the backtracking to help with the coordination
        # for testing, we want to see if the network implicitly coordinates the robots
        if train:
            times_visited = sensor_model.get_final_path().count(tuple(potential_next_loc)) + sensor_model.get_final_other_path().count(tuple(potential_next_loc))
        else:
            times_visited = sensor_model.get_final_path().count(tuple(potential_next_loc)) 

        if times_visited <= 0: # This means that the same action is allowed x + 1 times
            visited_before = False            
        else: 
            visited_before = True
        if valid_move == True and visited_before == False:
            break
        if counter > 10:
            break
   
    return action

# model here is the neural net
def greedy_planner(robot, sensor_model, neural_model, obs_occupied_oracle, curr_robot_positions, train, neural_net=False, oracle=False):
    actions = ['left', 'backward', 'right', 'forward']
    best_action_score = float('-inf')
    best_action = random.choice(actions)

    partial_info = [sensor_model.create_partial_info(False)]
    partial_info_binary_matrices = sensor_model.create_binary_matrices(partial_info)
    path_matrix = sensor_model.create_final_path_matrix(False)

    for action in actions:
        if robot.check_valid_move(action):
            # tuple is needed here for count()
            potential_next_loc = tuple(robot.get_action_loc(action))

            # only in data generation do we want the backtracking to help with the coordination
            # for testing, we want to see if the network implicitly coordinates the robots
            if train:
                times_visited = sensor_model.get_final_path().count(potential_next_loc) + sensor_model.get_final_other_path().count(potential_next_loc)
                # times_visited = sensor_model.get_final_path().count(potential_next_loc) + (potential_next_loc in sensor_model.get_final_other_path())
            else:
                times_visited = sensor_model.get_final_path().count(potential_next_loc) 
            
            # backtrack possibility
            if times_visited <= 1 and potential_next_loc not in curr_robot_positions: 
                if neural_net:
                    # We put partial_info and final_actions in a list because that's how those functions needed them in SensorModel
                    final_actions = [sensor_model.create_action_matrix(action, True)]
                    final_actions_binary_matrices = sensor_model.create_binary_matrices(final_actions)
                
                    input = NeuralNet.create_image(partial_info_binary_matrices, path_matrix, final_actions_binary_matrices)

                    # The unsqueeze adds an extra dimension at index 0 and the .float() is needed otherwise PyTorch will complain
                    # By unsqeezing, we add a batch dimension to the input, which is required by PyTorch: (n_samples, channels, height, width) 
                    input = input.unsqueeze(0).float()

                    action_score = neural_model(input).item()
                    
                else:
                    # oracle greedy knows where all the obstacles are
                    if oracle:
                        action_score = len(sensor_model.scan(potential_next_loc, obs_occupied_oracle, False)[0])
                    else:
                        scanned_unobs = sensor_model.scan(potential_next_loc, obs_occupied_oracle, False)
                        action_score = len(scanned_unobs[0]) + len(scanned_unobs[1])

                if action_score > best_action_score:
                    best_action_score = action_score
                    best_action = action
    
    # print("best_loc", robot.get_action_loc(best_action))
    # print("best action", best_action)
    # print('Path Debug: ', sensor_model.get_final_path().count(tuple(robot.get_action_loc(best_action)))
    # + sensor_model.get_final_other_path().count(tuple(robot.get_action_loc(best_action))))
    # print('Count Debug: '. len(sensor_model.get_final_path()))

    curr_robot_positions.add(tuple(robot.get_action_loc(best_action)))
    return best_action


def debug_greedy_planner(robot, sensor_model, neural_model, obs_occupied_oracle, train, debug_greedy_score, debug_network_score, neural_net=False, oracle=False):
    actions = ['left', 'backward', 'right', 'forward']
    best_action_score = float('-inf')
    best_action = random.choice(actions)
    net_action_score = 0
    greedy_action_score = 0

    partial_info = [sensor_model.create_partial_info(False)]
    partial_info_binary_matrices = sensor_model.create_binary_matrices(partial_info)
    path_matrix = sensor_model.create_final_path_matrix(False)

    for action in actions:
        if robot.check_valid_move(action):
            # tuple is needed here for count()
            potential_next_loc = tuple(robot.get_action_loc(action))

            # only in data generation do we want the backtracking to help with the coordination
            # for testing, we want to see if the network implicitly coordinates the robots
            if train:
                times_visited = sensor_model.get_final_path().count(potential_next_loc) + sensor_model.get_final_other_path().count(potential_next_loc)
            else:
                times_visited = sensor_model.get_final_path().count(potential_next_loc) 
            
            # backtrack possibility
            if times_visited <= 0: 
                

                # print("debug_network_score: ", debug_network_score)
                    
                # oracle greedy knows where all the obstacles are
                if oracle:
                    action_score = len(sensor_model.scan(potential_next_loc, obs_occupied_oracle, False)[0])

                else:
                    scanned_unobs = sensor_model.scan(potential_next_loc, obs_occupied_oracle, False)
                    action_score = len(scanned_unobs[0]) + len(scanned_unobs[1])

                if action_score > best_action_score:
                    best_action_score = action_score
                    best_action = action
    
    final_actions = [sensor_model.create_action_matrix(best_action, True)]
    final_actions_binary_matrices = sensor_model.create_binary_matrices(final_actions)
    input = NeuralNet.create_image(partial_info_binary_matrices, path_matrix, final_actions_binary_matrices)
    input = input.unsqueeze(0).float()
    net_action_score = neural_model(input).item()
    debug_network_score.append(net_action_score)
    debug_greedy_score.append(best_action_score)
    print("debug_greedy_score: ", debug_greedy_score)
    print("debug_network_score: ", net_action_score)

    return best_action, debug_network_score, debug_greedy_score




