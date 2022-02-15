from SensorModel import SensorModel
from Map import Map
from Robot import Robot
from Simulator import Simulator
import random
import time
from tqdm import tqdm
import pickle
import torch
import copy
import numpy as np
from util import get_random_loc, oracle_visualize, communicate, get_CONF, get_json_comp_conf


import matplotlib.pyplot as plt
import matplotlib.patches as patches


def generate_data_matrices(trials, steps, num_robots, planner_options, visualize, bounds, outfile, rollout=True):
    input_partial_info_binary_matrices = list()
    input_path_matrices = list()
    input_other_path_matrices = list()
    input_actions_binary_matrices = list()
    input_scores = list()

    for i in tqdm(range(trials)):
        # leaving this print statement out because tqdm takes care of progress
        # print("Trial: ", i)
        map = Map(bounds, 7, (), False)
        # unobs_occupied = copy.deepcopy(map.get_unobs_occupied())
        unobs_occupied = map.get_unobs_occupied()
        bots_starting_locs = list()

        # create robots
        robots = list()
        start_loc = get_random_loc(map, bounds)
        for _ in range(num_robots):
            # start_loc = get_random_loc(map, bounds)
            bot = Robot(start_loc[0], start_loc[1], bounds, map)
            robots.append(bot)
            bots_starting_locs.append(start_loc)

        for planner in planner_options: 
            obs_occupied_oracle = set()
            obs_free_oracle = set()

            # Bounds need to be an odd number for the action to always be in the middle
            for bot in robots:
                map = Map(bounds, 7, copy.deepcopy(unobs_occupied), True)
                sensor_model = SensorModel(bot, map)
                simulator = Simulator(map, bot, sensor_model, planner)
                start_loc = bot.get_start_loc()
                bot.set_loc(start_loc[0], start_loc[1])
                bot.add_map(map)
                bot.add_sensor_model(sensor_model)
                bot.add_simulator(simulator)
                # this adds the initial matrices to appropriate lists
                bot_simulator = bot.get_simulator()
                bot_simulator.initialize_data(bots_starting_locs, obs_occupied_oracle, generate_data=True)
                # this is needed incase any locations are scanned in the initial position
                obs_occupied_oracle = obs_occupied_oracle.union(bot_simulator.get_obs_occupied())
                obs_free_oracle = obs_free_oracle.union(bot_simulator.get_obs_free())
    
            for step in range(steps):
                # run multiple robots in same map
                for bot in robots:
                    simulator = bot.get_simulator()
                    sensor_model = bot.get_sensor_model()

                    if visualize:
                        simulator.visualize()

                    # false can be used as argument for neural_model here because we don't need mcts here
                    # obs_occupied_oracle is passed in so that scan() will calc the unique reward
                    # using true for train will make sure that the backtracking will consider other bot paths
                    simulator.run(False, curr_robot_positions=set(), obs_occupied_oracle=obs_occupied_oracle, train=False, generate_data=True)
                    # print("DEBUG PARTIAL IMAGE: ", sensor_model.get_final_partial_info()[-1])
                    # print("DEBUG SCORES: ", sensor_model.get_final_scores())
                    
                    # simulator.visualize(robots, step)


                    obs_occupied_oracle = obs_occupied_oracle.union(simulator.get_obs_occupied())
                    obs_free_oracle = obs_free_oracle.union(simulator.get_obs_free())
                    
                    communicate(robots, obs_occupied_oracle, obs_free_oracle)

                    if visualize: 
                        simulator.visualize() 

                # communicate(robots, obs_occupied_oracle, obs_free_oracle)
            
            # oracle_visualize(robots, bounds, map, planner)

            ### DATA MATRICES
            for bot in robots:
                sensor_model = bot.get_sensor_model()

                path_matricies = sensor_model.get_final_path_matrices()
                other_path_matricies = sensor_model.get_final_other_path_matrices()
                         
                final_partial_info = sensor_model.get_final_partial_info()
                partial_info_binary_matrices = sensor_model.create_binary_matrices(final_partial_info)

                final_actions = sensor_model.get_final_actions()
                final_actions_binary_matrices = sensor_model.create_binary_matrices(final_actions)

                final_scores = sensor_model.get_final_scores()

                input_path_matrices += path_matricies
                # print(path_matricies[0])
                # print(other_path_matricies[0])

                # input_other_path_matrices.append(path_matricies[0])s
                input_other_path_matrices += other_path_matricies
                # print("debug_input_path_matrices: ", input_path_matrices)
                # just for debugging
                # count = 0
                # for matrix in input_path_matrices:
                #     for row in matrix:
                #         for col in row:
                #             if col==1:
                #                 count+=1
                #     print("debug_input_path_matrices COUNT: ", count)
                #     count = 0
                #     print(matrix)

                input_partial_info_binary_matrices += partial_info_binary_matrices
                # print("debug_input_partial_info_binary_matrices: ", input_partial_info_binary_matrices)
                # count = 0
                # for matrix in input_partial_info_binary_matrices:
                #     # print("MATRIX: ", matrix)
                #     for matrix2 in matrix:
                #         # print("MATRIX2: ", matrix2)
                #         for row in matrix2:
                #             for col in row:
                #                 if col==1:
                #                     count+=1
                #         print("debug_input_partial_info_binary_matrices COUNT: ", count)
                #         count = 0
                #         print(matrix2)

                # print("images count: ", len(input_partial_info_binary_matrices))
                # for image in input_partial_info_binary_matrices:
                #     print("image: ", image)
                    # print("image: ", image[0])
                    # count = 0
                    # for row in image[0]:
                    #     for col in row:
                    #         if col == 1:
                    #             count += 1
                    # print("count: ", count)

                input_actions_binary_matrices +=  final_actions_binary_matrices
                input_scores += final_scores

            # rollout data is generated after the normal data of each map..
            # ..because when splitting the data at training, if the the normal data is created and the then the rollout..
            # .., the training and validation sets will have the same maps
            
            # print("data length: ", len(input_path_matrices))
            # print("data final_path_matrices: ", len(input_path_matrices))
            # print("final_other_path_matrices: ", len(input_other_path_matrices))
            # print(input_path_matrices[5])
            # print(input_other_path_matrices[5])

            
            if rollout:
                # print("Generating rollout data...")
                generate_data_rollout(input_path_matrices, input_other_path_matrices, input_partial_info_binary_matrices, input_actions_binary_matrices, input_scores, steps, num_robots, outfile)        
        
            # print("final_path_matrices: ", len(input_path_matrices))
            # print("final_partial_info_binary_matrices: ", len(input_partial_info_binary_matrices))
            # print("final_final_actions_binary_matrices", len(input_actions_binary_matrices))
            # print("final_final_scores: ", len(input_scores))

    # end = time.time()
    # time_taken = (end - start)/60
    # print("Iteration: {}, Planner: {}, Time taken: {:.3f}".format(i, planner, time_taken))

    print("final_path_matrices: ", len(input_path_matrices))
    print("final_partial_info_binary_matrices: ", len(input_partial_info_binary_matrices))
    print("final_final_actions_binary_matrices", len(input_actions_binary_matrices))
    print("final_final_scores: ", len(input_scores))

    print("Creating Torch tensors...")
    generate_tensor_images(input_path_matrices, input_partial_info_binary_matrices, input_actions_binary_matrices, input_scores, outfile)


def generate_data_rollout(input_path_matrices, input_other_path_matrices, input_partial_info_binary_matrices, input_actions_binary_matrices, input_scores, steps, num_robots, outfile):
    temp_input_partial_info_binary_matrices = list()
    temp_input_path_matrices = list()
    temp_input_other_path_matrices = list()
    temp_input_actions_binary_matrices = list()
    temp_input_scores = list()

    # "- ((steps*num_robots)+num_robots)" this is added because remember that rollout data is added..
    # to "input_partial_info_binary_matrices" so we need to start after the rollout data from the previous iteration
    index1 = len(input_partial_info_binary_matrices) - ((steps*num_robots)+num_robots)
    curr_robot =  1
    # the +1 in "boundary_increment" is because there is initial position matrix added to the arrays
    boundary_increment = (steps * curr_robot) + 1
    boundary = index1 + boundary_increment
    
    # the while loop is to make sure we don't iterate through the entire dataset to create..
    # ..rollout data because we are matching current data with future data
    while index1 <= (len(input_partial_info_binary_matrices) - steps//2):
        if curr_robot <= num_robots and index1 == boundary-(steps/5):
            curr_robot += 1
            # index1 becomes the previous boundary
            index1 = boundary
            # we move boundary forward by boundary increment
            boundary += boundary_increment
            
        temp_input_partial_info_binary_matrices.append(input_partial_info_binary_matrices[index1])
        # +1 because we don't want the same idx as index and -1 because it goes outside array otherwise
        index2 = random.randint(index1, boundary-2)
        # print("index2: ", index2)
        other_path_index = random.randint(index1, index2)
        # print("other_path_index: ", other_path_index)
        # print("len other path: ", len(input_other_path_matrices))
        # print("len path: ", len(input_path_matrices))
        # print()
        # debug
        # print()
        # print("index1: ", index1)
        # print("index2: ", index2)
        # print("boundary: ", boundary)
        # print()
        
        curr_input_path_matrix = input_path_matrices[index2]
        curr_other_path_matrix = input_other_path_matrices[other_path_index]

        # iterate over each row, col idx of the np array and modify path_matrix
        for irow, icol, in np.ndindex(curr_other_path_matrix.shape):
            if curr_other_path_matrix[irow, icol] == 1:
                curr_input_path_matrix[irow, icol]=1

        temp_input_path_matrices.append(curr_input_path_matrix)
        # these are just fillers so we get the correct idx in the next iteration
        temp_input_other_path_matrices.append(curr_other_path_matrix)

        temp_input_actions_binary_matrices.append(input_actions_binary_matrices[index2])
        temp_input_scores.append(input_scores[index2])

        index1 += 1

    input_partial_info_binary_matrices += temp_input_partial_info_binary_matrices
    input_path_matrices += temp_input_path_matrices
    input_other_path_matrices += temp_input_other_path_matrices
    input_actions_binary_matrices += temp_input_actions_binary_matrices
    input_scores += temp_input_scores

    # print("After rollout data: ")
    # print("final_path_matrices: ", len(input_path_matrices))
    # print("final_partial_info_binary_matrices: ", len(input_partial_info_binary_matrices))
    # print("final_final_actions_binary_matrices", len(input_actions_binary_matrices))
    # print("final_final_scores: ", len(input_scores))

    # generate_tensor_images() is being done in generate_data_matrices() itself
    # generate_tensor_images(input_path_matrices, input_partial_info_binary_matrices, input_actions_binary_matrices, input_scores, outfile)


def generate_tensor_images(path_matricies, partial_info_binary_matrices, final_actions_binary_matrices, final_scores, outfile): 
    data = list()

    for i in tqdm(range(len(partial_info_binary_matrices))):
        image = list()

        for partial_info in partial_info_binary_matrices[i]:
            image.append(partial_info)

        image.append(path_matricies[i])

        for action in final_actions_binary_matrices[i]:
            image.append(action)

        # this is needed, because torch complains otherwise that converting a list is too slow
        # it's better to use a np array because of the way a np array is stored in memory (contiguous)
        image = np.array(image)
        
        data.append([torch.IntTensor(image), final_scores[i]])

    # pickle progress
    print("Pickling started!")
    outfile_tensor_images = open(outfile, 'wb')
    pickle.dump(data, outfile_tensor_images)
    outfile_tensor_images.close()
    print("Pickling done!")


if __name__ == "__main__":

    CONF = get_CONF()
    json_comp_conf = get_json_comp_conf()

    # for pickling
    # outfile_tensor_images = '/home/kavi/thesis/pickles/data_21x21_circles_random_greedyno_r4_t2000_s25_rollout_diffstartloc_otherpathmix'
    datafile = "data_21x21_circles_random_greedyno_r4_t2000_s25_rolloutotherpath_samestartloc_commscorrected"
    outfile_tensor_images = CONF[json_comp_conf]["pickle_path"] + datafile
    
    # generate data
    print("Generating matrices")
    planner_options = ["random_fullcomm", "greedy_fullcomm"]
    generate_data_matrices(trials=2000, steps=25, num_robots=4, planner_options=planner_options, visualize=False, bounds=[21, 21], outfile=outfile_tensor_images)
    