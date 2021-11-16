from SensorModel import SensorModel
from Map import Map
from Robot import Robot
from Simulator import Simulator
import random
import time
from tqdm import tqdm
import pickle
import torch

# rollout produces the unique data needed for mcts rollout
# this is done because in rollout, belief map stays the same even though path and actions change
def generate_data_matrices(trials, steps, planner_options, visualize, outfile, rollout=True):
    input_partial_info_binary_matrices = list()
    input_path_matrices = list()
    input_actions_binary_matrices = list()
    input_scores = list()

    for i in tqdm(range(trials)):
        # leaving this print statement out because tqdm takes care of progress
        # print("Trial: ", i)
        for planner in planner_options: 
            start = time.time()
            # Bounds need to be an odd number for the action to always be in the middle
            bounds = [21, 21]
            map = Map(bounds, 6, [])

            # Selects random starting locations for the robot
            # We can't use the exact bounds (need -1) due to the limits we create in checking valid location functions
            valid_starting_loc = False
            while not valid_starting_loc:
                x = random.randint(0, bounds[0]-1)
                y = random.randint(0, bounds[0]-1)
                valid_starting_loc = map.check_loc(x, y) 

            robot = Robot(x, y, bounds, map)
            sensor_model = SensorModel(robot, map)
            
            simulator = Simulator(map, robot, sensor_model, planner)
            if visualize:
                simulator.visualize()
            simulator.run(steps, False)
            
            if visualize: 
                simulator.visualize()
            
            ### TRAINING DATA
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

            end = time.time()
            time_taken = (end - start)/60
            # print("Iteration: {}, Planner: {}, Time taken: {:.3f}".format(i, planner, time_taken))
    
    
    print("final_path_matrices: ", len(input_path_matrices))
    print("final_partial_info_binary_matrices: ", len(input_partial_info_binary_matrices))
    print("final_final_actions_binary_matrices", len(input_actions_binary_matrices))
    print("final_final_scores: ", len(input_scores))

    if rollout:
        generate_data_rollout(input_path_matrices, input_partial_info_binary_matrices, input_actions_binary_matrices, input_scores, outfile)
    else: 
        generate_tensor_images(input_path_matrices, input_partial_info_binary_matrices, input_actions_binary_matrices, input_scores, outfile)


def generate_data_rollout(input_path_matrices, input_partial_info_binary_matrices, input_actions_binary_matrices, input_scores, outfile):
    temp_input_partial_info_binary_matrices = list()
    temp_input_path_matrices = list()
    temp_input_actions_binary_matrices = list()
    temp_input_scores = list()

    # integer divide by two because we don't want to double the dataset size, but just a decent amount of samples
    # -5 because index 2 has to choose values ahead of index1
    for index1 in tqdm(range((len(input_partial_info_binary_matrices)-5)//(4//3))):
        temp_input_partial_info_binary_matrices.append(input_partial_info_binary_matrices[index1])
        # +1 because we don't want the same idx as index and -1 because it goes outside array otherwise
        index2 = random.randint(index1+1, len(input_partial_info_binary_matrices)-1)
        
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

    generate_tensor_images(input_path_matrices, input_partial_info_binary_matrices, input_actions_binary_matrices, input_scores, outfile)


def generate_tensor_images(path_matricies, partial_info_binary_matrices, final_actions_binary_matrices, final_scores, outfile): 
    data = list()

    for i in tqdm(range(len(partial_info_binary_matrices))):
        image = list()

        for partial_info in partial_info_binary_matrices[i]:
            image.append(partial_info)

        image.append(path_matricies[i])

        for action in final_actions_binary_matrices[i]:
            image.append(action)
        
        data.append([torch.IntTensor(image), final_scores[i]])

    # pickle progress
    print("Pickling started!")
    outfile_tensor_images = open(outfile, 'wb')
    pickle.dump(data, outfile_tensor_images)
    outfile_tensor_images.close()
    print("Pickling done!")


if __name__ == "__main__":

    # for pickling
    outfile_tensor_images = '/home/kavi/thesis/pickles/data_21x21_random_greedyo_greedyno_t1200_s150_norollout'
    
    # generate data
    print("Generating matrices")
    planner_options = ["random", "greedy-o", "greedy-no"]
    generate_data_matrices(trials=1200, steps=150, planner_options=planner_options, visualize=False, outfile=outfile_tensor_images, rollout=False)
    print()
    