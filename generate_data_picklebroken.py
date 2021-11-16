from SensorModel import SensorModel
from Map import Map
from Robot import Robot
from Simulator import Simulator
import random
import time
from tqdm import tqdm
import pickle
import torch

def generate_data_matrices(outfile1, outfile2, outfile3, outfile4):
    input_partial_info_binary_matrices = list()
    input_path_matrices = list()
    input_actions_binary_matrices = list()
    input_scores = list()

    planner_options = ["random", "greedy-o", "greedy-no"]
    # planner_options = ["random"]
    # trials = 600
    trials = 6
    # steps = 150
    steps = 15
    visualize = False
    train = True
    
    for i in tqdm(range(trials)):
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
            
            if train:
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
    
    if train:
        print("final_path_matrices: ", len(input_path_matrices))
        print("final_partial_info_binary_matrices: ", len(input_partial_info_binary_matrices))
        print("final_final_actions_binary_matrices", len(input_actions_binary_matrices))
        print("final_final_scores: ", len(input_scores))

        # pickle all the data before rollout
        print("Pickling started!")
        outfile1 = open(file1, 'wb')
        pickle.dump(input_path_matrices, outfile1)
        outfile1.close()
        outfile2 = open(file2, 'wb')
        pickle.dump(input_partial_info_binary_matrices, outfile2)
        outfile2.close()
        outfile3 = open(file3, 'wb')
        pickle.dump(input_actions_binary_matrices, outfile3)
        outfile3.close()
        outfile4 = open(file4, 'wb')
        pickle.dump(input_scores, outfile4)
        outfile4.close()
        print("Pickling done!")


def generate_data_rollout(file1, file2, file3, file4, outfile1, outfile2, outfile3, outfile4):
    # unpickle scores
    print("Unpickling started!")
    infile1 = open(file1, 'rb')
    input_path_matrices = pickle.load(infile1)
    infile1.close()
    infile2 = open(file2, 'rb')
    input_partial_info_binary_matrices = pickle.load(infile2)
    infile2.close()
    infile3 = open(file3, 'rb')
    input_actions_binary_matrices = pickle.load(infile3)
    infile3.close()
    infile4 = open(file4, 'rb')
    input_scores = pickle.load(infile4)
    infile4.close()
    print("Unpickling done!")

    temp_input_partial_info_binary_matrices = list()
    temp_input_path_matrices = list()
    temp_input_actions_binary_matrices = list()
    temp_input_scores = list()

    # visited = list() 
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

    # pickle all the data after rollout
    outfile_rollout1 = open(outfile1, 'wb')
    pickle.dump(input_path_matrices, outfile_rollout1)
    outfile_rollout1.close()
    outfile_rollout2 = open(outfile2, 'wb')
    pickle.dump(input_partial_info_binary_matrices, outfile_rollout2)
    outfile_rollout2.close()
    outfile_rollout3 = open(outfile3, 'wb')
    pickle.dump(input_actions_binary_matrices, outfile_rollout3)
    outfile_rollout3.close()
    outfile_rollout4 = open(outfile4, 'wb')
    pickle.dump(input_scores, outfile_rollout4)
    outfile_rollout4.close()


def generate_tensor_images(file1, file2, file3, file4, outfile): 
    # unpickle scores
    print("Unpickling started!")
    infile_rollout1 = open(file1, 'rb')
    path_matricies = pickle.load(infile_rollout1)
    infile_rollout1.close()
    infile_rollout2 = open(file2, 'rb')
    partial_info_binary_matrices = pickle.load(infile_rollout2)
    infile_rollout2.close()
    infile_rollout3 = open(file3, 'rb')
    final_actions_binary_matrices = pickle.load(infile_rollout3)
    infile_rollout3.close()
    infile_rollout4 = open(file4, 'rb')
    final_scores = pickle.load(infile_rollout4)
    infile_rollout4.close()
    print("Unpickling done!")

    data = list()

    for i in tqdm(range(len(partial_info_binary_matrices))):
        image = list()

        for partial_info in partial_info_binary_matrices[i]:
            image.append(partial_info)

        image.append(path_matricies[i])

        for action in final_actions_binary_matrices[i]:
            image.append(action)
        
        print(image)
        
        data.append([torch.IntTensor(image), final_scores[i]])

    # pickle progress
    print("Pickling started!")
    outfile_tensor_images = open(outfile, 'wb')
    pickle.dump(data, outfile_tensor_images)
    outfile_tensor_images.close()
    print("Pickling done!")


if __name__ == "__main__":

    # pickle directories
    # file1 = open('/home/kavi/thesis/pickles/input_path_matrices','wb')
    # file2 = open('/home/kavi/thesis/pickles/input_partial_info_binary_matrices','wb')
    # file3 = open('/home/kavi/thesis/pickles/input_actions_binary_matrices','wb')
    # file4 = open('/home/kavi/thesis/pickles/input_scores','wb')

    file1 = '/home/kavi/thesis/pickles/input_path_matrices'
    file2 = '/home/kavi/thesis/pickles/input_partial_info_binary_matrices'
    file3 = '/home/kavi/thesis/pickles/input_actions_binary_matrices'
    file4 = '/home/kavi/thesis/pickles/input_scores'


    # file1_rollout = open('/home/kavi/thesis/pickles/input_path_matrices_rollout','wb')
    # file2_rollout = open('/home/kavi/thesis/pickles/input_partial_info_binary_matrices_rollout','wb')
    # file3_rollout = open('/home/kavi/thesis/pickles/input_actions_binary_matrices_rollout','wb')
    # file4_rollout = open('/home/kavi/thesis/pickles/input_scores_rollout','wb')

    file1_rollout = '/home/kavi/thesis/pickles/input_path_matrices_rollout'
    file2_rollout = '/home/kavi/thesis/pickles/input_partial_info_binary_matrices_rollout'
    file3_rollout = '/home/kavi/thesis/pickles/input_actions_binary_matrices_rollout'
    file4_rollout = '/home/kavi/thesis/pickles/input_scores_rollout'

    outfile_tensor_images = '/home/kavi/thesis/pickles/data_21x21_random_greedyo_greedyno_t600_s150'
    
    # generate data
    print("Generating matrices")
    generate_data_matrices(file1, file1, file1, file1)
    print()
    print("Generating rollout data")
    generate_data_rollout(file1, file1, file1, file1, file1_rollout, file2_rollout, file3_rollout, file4_rollout)
    print()
    print("Generating images")
    generate_tensor_images(file1_rollout, file2_rollout, file3_rollout, file4_rollout, outfile_tensor_images)

    