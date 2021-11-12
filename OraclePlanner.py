import random
import torch
import NeuralNet

def random_planner(robot, sensor_model):
    actions = ['left', 'right', 'backward', 'forward']
    valid_move = False # Checks if the pixel is free
    visited_before = True # Check if the pixel has been visited before
    action = random.choice(actions)
   
    counter = 0
    while True:
        counter += 1
        action = random.choice(actions)
        valid_move = robot.check_valid_move(action) 
        times_visited = sensor_model.get_final_path().count(tuple(robot.get_action_loc(action)))
        robot_loc_debug = robot.get_loc()
        action_loc_debug = robot.get_action_loc(action)
        if times_visited > 1: # This means that the same action is allowed x + 1 times
            visited_before = True
        else: 
            visited_before = False
        if valid_move == True and visited_before == False:
            break
        if counter > 10:
            break
   
    return action

# model here is the neural net
def greedy_planner(robot, sensor_model, neural_model, neural_net=False, oracle=False):
    actions = ['left', 'backward', 'right', 'forward']
    best_action_score = float('-inf')
    best_action = random.choice(actions)

    # load neural net with weights and set to forward prop only
    # model = NeuralNet.Net(map.get_bounds())
    # model.load_state_dict(torch.load("/home/kavi/thesis/neural_net_weights/circles_random_21x21_epoch2_mctsrolloutdata2"))
    # model.eval()

    partial_info = [sensor_model.create_partial_info(False)]
    partial_info_binary_matrices = sensor_model.create_binary_matrices(partial_info)
    path_matrix = sensor_model.create_final_path_matrix(False)

    for action in actions:
        if robot.check_valid_move(action):
            # tuple is needed here for count()
            potential_next_loc = tuple(robot.get_action_loc(action))
            times_visited = sensor_model.get_final_path().count(potential_next_loc)
            
            
            # backtrack possibility
            if times_visited <= 1: 
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
                        action_score = len(sensor_model.scan(potential_next_loc, False)[0])
                    else:
                        scanned_unobs = sensor_model.scan(potential_next_loc, False)
                        action_score = len(scanned_unobs[0]) + len(scanned_unobs[1])

                if action_score > best_action_score:
                    best_action_score = action_score
                    best_action = action

    # print('Path Debug: ', sensor_model.get_final_path().count(tuple(robot.get_action_loc(best_action))))
    # print('Count Debug: '. len(sensor_model.get_final_path()))
    return best_action




