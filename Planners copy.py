import random
import NeuralNet
from utils import backtrack_count

def random_planner(bot, curr_robot_positions):
    actions = ['left', 'right', 'backward', 'forward']
    valid_move = False # checks if the pixel is free
    visited_before = True # check if the pixel has been visited before
    action = random.choice(actions)
    bot_belief_map = bot.get_belief_map()
    curr_bot_loc = bot.get_loc()
   
    counter = 0
    while True:
        counter += 1
        action = random.choice(actions)
        valid_move = bot_belief_map.is_valid_action(action, curr_bot_loc)
        potential_loc = bot_belief_map.get_action_loc(action, curr_bot_loc)
        
        exec_paths = bot.get_exec_paths()
        other_exec_paths = bot.get_other_exec_paths()

        if backtrack_count(exec_paths, other_exec_paths, potential_loc) <= 1 \
                and (potential_loc not in curr_robot_positions):
            visited_before = False            
        else: 
            visited_before = True
        if valid_move == True and visited_before == False:
            break
        if counter > 10: # just incase the while loop gets caught in inifite loop
            break
   
    return action

# model here is the neural net
def greedy_planner(robot, sensor_model, neural_model, curr_robot_positions, neural_net=False, device=False):
    actions = ['left', 'backward', 'right', 'forward']
    best_action_score = float('-inf')
    best_action = random.choice(actions)

    partial_info = [sensor_model.create_partial_info(False)]
    partial_info_binary_matrices = sensor_model.create_binary_matrices(partial_info)
    path_matrix = sensor_model.create_final_path_matrix(False)

    for action in actions:
        if robot.check_valid_move(action):
            # tuple is needed here for count()
            potential_loc = tuple(robot.get_action_loc(action))
            
            # backtrack possibility
            if backtrack_count(exec_path, other_exec_path, potential_loc) <= 1 \
                and (potential_loc not in curr_robot_positions):
                if neural_net:
                    # We put partial_info and final_actions in a list because that's how those functions needed them in SensorModel
                    final_actions = [sensor_model.create_action_matrix(action, True)]
                    final_actions_binary_matrices = sensor_model.create_binary_matrices(final_actions)
                
                    input = NeuralNet.create_image(partial_info_binary_matrices, path_matrix, final_actions_binary_matrices)

                    # The unsqueeze adds an extra dimension at index 0 and the .float() is needed otherwise PyTorch will complain
                    # By unsqeezing, we add a batch dimension to the input, which is required by PyTorch: (n_samples, channels, height, width) 
                    input = input.unsqueeze(0).float().to(device)

                    action_score = neural_model(input).item()
                    
                else:
                    scanned_items = sensor_model.scan(potential_loc, False)
                    # oracle greedy knows where all the obstacles are
                    # if oracle:
                        # action_score = len(scanned_items[0])
                    # else:
                        # action_score = len(scanned_items[0]) + len(scanned_items[1])
                    
                    action_score = len(scanned_items[0]) + len(scanned_items[1])


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





