from SensorModel import SensorModel
from Map import Map
from Robot import Robot
from Simulator import Simulator
import NeuralNet
import random
import time

if __name__ == "__main__":

    input_partial_info_binary_matrices = list()
    input_path_matrices = list()
    input_actions_binary_matrices = list()
    input_scores = list()

    # planner_options = ["random", "greedy"]
    planner_options = ["random"]
    for i in range(50):
        for planner in planner_options: 
            start = time.time()
            # Bounds need to be an odd number for the action to always be in the middle
            bounds = [21, 21]
            map = Map(bounds, 18, ())

            # Selects random starting locations for the robot
            # We can't use 111 due to the limits we create in checking valid location functions
            valid_starting_loc = False
            while not valid_starting_loc:
                x = random.randint(0, bounds[0])
                y = random.randint(0, bounds[1])
                valid_starting_loc = map.check_loc(x, y) 

            robot = Robot(x, y, bounds, map)
            sensor_model = SensorModel(robot, map)
            
            simulator = Simulator(map, robot, sensor_model, planner)
            simulator.run(50, False)

            # simulator.visualize()
            
            ### Training data
            path_matricies = sensor_model.get_final_path_matrices()

            final_partial_info = sensor_model.get_final_partial_info()
            partial_info_binary_matrices = sensor_model.create_binary_matrices(final_partial_info)

            final_actions = sensor_model.get_final_actions()
            final_actions_binary_matrices = sensor_model.create_binary_matrices(final_actions)

            final_scores = sensor_model.get_final_scores()

            input_path_matrices = input_path_matrices + path_matricies
            input_partial_info_binary_matrices = input_partial_info_binary_matrices + partial_info_binary_matrices
            input_actions_binary_matrices = input_actions_binary_matrices + final_actions_binary_matrices
            input_scores = input_scores + final_scores
            # print(input_scores)

            ### Add training data to account for rollout
            # need to create new lists for these
            # in the end, i need to concatenate the final lists
            

            end = time.time()
            time_taken = (end - start)/60
            print("Iteration: {}, Planner: {}, Time taken: {:.3f}".format(i, planner, time_taken))

    print("final_path_matrices: ", len(input_path_matrices))
    print("final_partial_info_binary_matrices: ", len(input_partial_info_binary_matrices))
    print("final_final_actions_binary_matrices", len(input_actions_binary_matrices))
    print("final_final_scores: ", len(input_scores))

    ### ADD DATA FOR ROLLOUT

    temp_input_partial_info_binary_matrices = list()
    temp_input_path_matrices = list()
    temp_input_actions_binary_matrices = list()
    temp_input_scores = list()

    visited = list() 
    # integer divide by two because we don't want to double the dataset size, but just a decent amount of samples
    for _ in range(len(input_partial_info_binary_matrices)//2):
        # -2 here because if not the the randint() ends up like randint(1,0), where first value is higher
        index1 = random.randint(0, len(input_partial_info_binary_matrices)-2)
        print("length: ", len(input_partial_info_binary_matrices))
        if index1 not in visited:
            visited.append(index1)
            temp_input_partial_info_binary_matrices.append(input_partial_info_binary_matrices[index1])
            print("index1: ", index1)
            # +1 because we don't want the same idx as index and -1 because it goes outside array otherwise
            index2 = random.randint(index1+1, len(input_partial_info_binary_matrices)-1)
            
            print("index2: ", index2)
            temp_input_path_matrices.append(input_path_matrices[index2])
            temp_input_actions_binary_matrices.append(input_actions_binary_matrices[index2])
            temp_input_scores.append(input_scores[index2])

    input_partial_info_binary_matrices += temp_input_partial_info_binary_matrices
    input_path_matrices += temp_input_path_matrices
    input_actions_binary_matrices += temp_input_actions_binary_matrices
    input_scores += temp_input_scores

    print("final_path_matrices: ", len(input_path_matrices))
    print("final_partial_info_binary_matrices: ", len(input_partial_info_binary_matrices))
    print("final_final_actions_binary_matrices", len(input_actions_binary_matrices))
    print("final_final_scores: ", len(input_scores))

    ### Train network
    # data = NeuralNet.datasetGenerator(input_partial_info_binary_matrices, input_path_matrices, input_actions_binary_matrices, input_scores)
    # NeuralNet.runNetwork(data, bounds)



