import random
import torch
import NeuralNet

def random_planner(robot, sensor_model):
    actions = ['left', 'right', 'backward', 'forward']
    valid_move = False # Checks if the pixel is free
    visited_before = True # Check if the pixel has been visited before
    action = random.choice(actions)
    
    ### Old code used where a check wasn't done to see if it had visited the same location
    # while not valid_move and tuple(robot.get_action_loc(action)) not in sensor_model.get_final_path():
    # while not valid_move:
    #     action = random.choice(actions)
    #     valid_move = robot.check_valid_move(action) 
    #     while not loc:
    #         # loc = tuple(robot.get_action_loc(action)) not in sensor_model.get_final_path()
    #         loc = sensor_model.get_final_path().count(robot.get_action_loc(action)) < 2
    counter = 0

    while True:
        counter += 1
        action = random.choice(actions)
        valid_move = robot.check_valid_move(action) 
        times_visited = sensor_model.get_final_path().count(tuple(robot.get_action_loc(action)))
        robot_loc_debug = robot.get_loc()
        action_loc_debug = robot.get_action_loc(action)
        if times_visited > 6: # This means that the same action is allowed x + 1 times
            visited_before = True
        else: 
            visited_before = False
        if valid_move == True and visited_before == False:
            break
        if counter > 100:
            break
   
    return action

def greedy_planner(robot, sensor_model, map, neural_net=False):
    actions = ['left', 'right', 'backward', 'forward']
    # actions = ['backward', 'forward']
    # best_action = random_planner(robot, sensor_model)
    # best_action = None
    best_action = random.choice(actions)
    best_action_score = float('-inf')
    counter = 0

    model = NeuralNet.Net(map.get_bounds())
    model.load_state_dict(torch.load("/home/kavi/thesis/neural_net_weights/circles_random_21x21"))
    # model.load_state_dict(torch.load("/home/kavi/thesis/neural_net_weights/circles_random_21x21_epoch2"))
    model.eval()

    partial_info = [sensor_model.create_partial_info(False)]
    partial_info_binary_matrices = sensor_model.create_binary_matrices(partial_info)

    path_matrix = sensor_model.create_final_path_matrix(False)

    # while counter == 0:
    # while best_action == None:
    while True:
        for action in actions:
            counter += 1
            times_visited = sensor_model.get_final_path().count(tuple(robot.get_action_loc(action)))
            robot_loc_debug = robot.get_loc()
            action_loc_debug = robot.get_action_loc(action)
            # if robot.check_valid_move(action) and times_visited < 1: # This means times_visited - 1 is allowed e.g. times_visited < 1 means 0 times allowed
            if robot.check_valid_move(action):
                if times_visited < 6: # This means times_visited is allowed e.g. times_visited < 1 means 1 times allowed to be in list
                    temp_robot_loc = robot.get_action_loc(action)
                    if neural_net:
                        # We put partial_info and final_actions in a list because that's how those functions needed them in SensorModel
                        final_actions = [sensor_model.create_action_matrix(action, True)]
                        final_actions_binary_matrices = sensor_model.create_binary_matrices(final_actions)
                    
                        input = NeuralNet.create_image(partial_info_binary_matrices, path_matrix, final_actions_binary_matrices)

                        # The unsqueeze adds an extra dimension at index 0 and the .float() is needed otherwise PyTorch will complain
                        # By unsqeezing, we add a batch dimension to the input, which is required by PyTorch: (n_samples, channels, height, width) 
                        input = input.unsqueeze(0).float()

                        action_score = model(input).item()
                        # action_score = -1
                        # print(action_score)
                        
                    else:
                        # counter += 1
                        action_score = len(sensor_model.scan(temp_robot_loc, False)[0])
                    if action_score > best_action_score:
                        best_action_score = action_score
                        best_action = action

        if counter > 100:
            break

    return best_action

# def greedy_planner(robot, sensor_model, map, neural_net=False):
#     actions = ['left', 'right', 'backward', 'forward']
#     best_action = random_planner(robot)
#     best_action_score = 0

#     model = NeuralNet.Net(map.get_bounds())
#     model.load_state_dict(torch.load("/home/kavi/thesis/neural_net_weights/circles_random_21x21"))
#     # model.load_state_dict(torch.load("/home/kavi/thesis/neural_net_weights/circles_random_21x21_epoch2"))
#     model.eval()

#     for action in actions:
#         if robot.check_valid_move(action):
#             temp_robot_loc = robot.get_action_loc(action)
#             if neural_net:
#                 # We put partial_info and final_actions in a list because that's how those functions needed them in SensorModel
#                 partial_info = [sensor_model.create_partial_info(False)]
#                 partial_info_binary_matrices = sensor_model.create_binary_matrices(partial_info)

#                 path_matrix = sensor_model.create_final_path_matrix(False)

#                 final_actions = [sensor_model.create_action_matrix(action, True)]
#                 final_actions_binary_matrices = sensor_model.create_binary_matrices(final_actions)
            
#                 input = NeuralNet.create_image(partial_info_binary_matrices, path_matrix, final_actions_binary_matrices)

#                 # The unsqueeze adds an extra dimension at index 0 and the .float() is needed otherwise PyTorch will complain
#                 # By unsqeezing, we add a batch dimension to the input, which is required by PyTorch: (n_samples, channels, height, width) 
#                 input = input.unsqueeze(0).float()

#                 action_score = model(input).item()
#                 # print(action_score)
#             else:
#                 action_score = len(sensor_model.scan(temp_robot_loc, False)[0])
#             if action_score > best_action_score:
#                 best_action_score = action_score
#                 best_action = action

#     return best_action


