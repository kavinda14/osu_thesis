import random
import torch

def random_planner(robot):
    actions = ['left', 'right', 'backward', 'forward']
    valid_move = False
    action = ''

    while not valid_move:
        action = random.choice(actions)
        valid_move = robot.check_valid_move(action) 
    
    return action

def greedy_planner(robot, sensor_model, neural_net=False):
    actions = ['left', 'right', 'backward', 'forward']
    best_action = random_planner(robot)
    best_action_score = 0

    for action in actions:
        if robot.check_valid_move(action):
            temp_robot_loc = robot.get_action_loc(action)
            if neural_net:
                partial_info = sensor_model.create_partial_info()
                partial_info_binary_matrices = sensor_model.create_binary_matrices(partial_info)

                path_matrix = sensor_model.create_final_path_matrix(False)

                final_actions = sensor_model.create_action_matrix(action)
                final_actions_binary_matrices = sensor_model.create_binary_matrices(final_actions)

                input = list()

                for partial_info in partial_info_binary_matrices[0]:
                    input.append(partial_info)

                input.append(path_matrix)

                for action in final_actions_binary_matrices[0]:
                    input.append(action)


                input = torch.IntTensor(input)
                print(input)
                # actions_score = # Send input through neural network to get the score
            else:
                action_score = len(sensor_model.scan(temp_robot_loc, False)[0])
            if action_score > best_action_score:
                best_action_score = action_score
                best_action = action

    return best_action


