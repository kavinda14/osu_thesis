import random
import torch
import NeuralNet

def random_planner(robot):
    actions = ['left', 'right', 'backward', 'forward']
    valid_move = False
    action = ''

    while not valid_move:
        action = random.choice(actions)
        valid_move = robot.check_valid_move(action) 
    
    return action

def greedy_planner(robot, sensor_model, map, neural_net=False):
    actions = ['left', 'right', 'backward', 'forward']
    best_action = random_planner(robot)
    best_action_score = 0

    model = NeuralNet.Net(map.get_bounds())
    model.load_state_dict(torch.load("/home/kavi/thesis/neural_net_weights/circles"))
    model.eval()

    for action in actions:
        if robot.check_valid_move(action):
            temp_robot_loc = robot.get_action_loc(action)
            if neural_net:
                # We put partial_info and final_actions in a list because that's how those functions needed them in SensorModel
                partial_info = [sensor_model.create_partial_info(False)]
                partial_info_binary_matrices = sensor_model.create_binary_matrices(partial_info)

                path_matrix = sensor_model.create_final_path_matrix(False)

                final_actions = [sensor_model.create_action_matrix(action, True)]
                final_actions_binary_matrices = sensor_model.create_binary_matrices(final_actions)
            
                input = NeuralNet.create_image(partial_info_binary_matrices, path_matrix, final_actions_binary_matrices)

                # The unsqueeze adds an extra dimension at index 0 and the .float() is needed otherwise PyTorch will complain
                # By unsqeezing, we add a batch dimension to the input, which is required by PyTorch: (n_samples, channels, height, width) 
                input = input.unsqueeze(0).float()

                action_score = model(input).item()
                # print(action_score)
            else:
                action_score = len(sensor_model.scan(temp_robot_loc, False)[0])
            if action_score > best_action_score:
                best_action_score = action_score
                best_action = action

    return best_action


