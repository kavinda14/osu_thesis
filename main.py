import matplotlib.pyplot as plt
import matplotlib.patches as patches
from GroundTruthMap import GroundTruthMap
from utils import get_random_loc, get_CONF, get_json_comp_conf
import torch
from NeuralNet import Net
from time import time
from tqdm import tqdm
from BeliefMap import BeliefMap
from Robot import Robot
from SensorModel import SensorModel
from Simulator import Simulator
from Planners import RandomPlanner, CellCountPlanner
from copy import deepcopy
import numpy as np
from random import random, randint
import pickle

def vis_map(robots, bounds, map):
    plt.xlim(0, bounds[0])
    plt.ylim(0, bounds[1])

    ax = plt.gca()
    ax.set_aspect('equal', 'box')

    # this has to be done before the bot for loop to avoid red patches
    # ..going over the other obs_occupied patches
    try: # since only BeliefMap has unknown_locs, we can use try except
        for spot in map.get_unknown_locs():
            hole = patches.Rectangle(spot, 1, 1, facecolor='black')
            ax.add_patch(hole)
    except:
        # color all occupied locs before putting specific bot colors on them (to identify which bot discovered what)
        occupied_locs = map.get_occupied_locs()
        for spot in occupied_locs:
            hole = patches.Rectangle(spot, 1, 1, facecolor='red')
            ax.add_patch(hole)

    # get all the observed locations from all robots
    free_locs = set()
    for bot in robots:
        bot_belief_map = bot.get_belief_map()
     
        free_locs = free_locs.union(bot_belief_map.get_free_locs())
        occupied_locs = bot_belief_map.get_occupied_locs()

        bot_color = bot.get_color()

        # plot robot
        robot_x = bot.get_loc()[0] + 0.5
        robot_y = bot.get_loc()[1] + 0.5
        plt.scatter(robot_x, robot_y, color=bot_color, zorder=5)

        # plot robot path
        x_values = list()
        y_values = list()
        for path in bot.get_exec_path():
            x_values.append(path[0] + 0.5)
            y_values.append(path[1] + 0.5)
        plt.plot(x_values, y_values, color=bot_color)

        # plot occupied_locs
        # this is in the loop so that we can use diff colors for each robot's occ cells 
        for spot in occupied_locs:
            hole = patches.Rectangle(spot, 1, 1, facecolor=bot_color)
            ax.add_patch(hole)

    # plot free_locs
    for spot in free_locs:
        hole = patches.Rectangle(spot, 1, 1, facecolor='white')
        ax.add_patch(hole)

    plt.show()

def plot_scores(saved_scores):
    x_pos = np.arange(1, len(saved_scores)+1) # aligns xaxis labels properly
    plt.figure(figsize=(10, 7))
    plt.boxplot([scores for scores in saved_scores.values()])
    plt.xticks(x_pos, [planner for planner in saved_scores.keys()], rotation=25)
    # plt.title(weight_file+"_trials:"+str(trials)+"_steps:"+str(steps))
    plt.tight_layout()
    plt.show()

def get_neural_model(CONF, json_comp_conf, bounds):
    weight_file = "circles_21x21_epoch1_random_greedyno_r4_t2000_s35_rolloutotherpath_samestartloc_obsdensity18"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)
    neural_model = Net(bounds).to(device)
    neural_model.load_state_dict(torch.load(
        CONF[json_comp_conf]["neural_net_weights_path"]+weight_file))
    neural_model.eval()
    
    return neural_model, device

def get_robots(num_robots, belief_map, ground_truth_map, robot_start_loc):
    robots = set()
    for i in range(num_robots):
        start_loc = robot_start_loc[i] # use this when testing if paths being communicated properly
        belief_map_copy = deepcopy(belief_map) # to make sure that each robot has a diff belief map object
        # bot = Robot(robot_start_loc[0], robot_start_loc[1], belief_map_copy)
        bot = Robot(start_loc[0], start_loc[1], belief_map_copy)
        sensor_model = SensorModel(bot, belief_map)
        simulator = Simulator(belief_map_copy, ground_truth_map, bot, sensor_model, generate_data=False)
        bot.set_sensor_model(sensor_model)
        bot.set_simulator(simulator)
        robots.add(bot)

    return robots

def generate_binary_matrices(robots, path_matrices, comm_path_matrices, partial_info_binary_matrices, 
                            action_binary_matrices, scores, total_steps, num_robots, rollout):
    for bot in robots:
        sensor_model = bot.get_sensor_model()

        # path matricies
        bot_path_matrices = sensor_model.get_path_matrices()
        bot_comm_path_matrices = sensor_model.get_comm_path_matrices()

        # partial info matricies
        bot_partial_info_matricies = sensor_model.get_partial_info_matrices()
        bot_partial_info_binary_matrices = sensor_model.create_binary_matrices(
            bot_partial_info_matricies)

        # action matrices
        bot_actions_matrices = sensor_model.get_action_matrices()
        bot_action_binary_matrices = sensor_model.create_binary_matrices(
            bot_actions_matrices)

        # scores
        bot_scores = bot.get_simulator().get_scores()

        path_matrices += bot_path_matrices

        comm_path_matrices += bot_comm_path_matrices
        # print("debug_path_matrices:")
        # print(path_matrices)
        # # just for debugging
        # count = 0
        # for matrix in path_matrices:
        #     for row in matrix:
        #         for col in row:
        #             if col==1:
        #                 count+=1
        #     print("debug_path_matrices COUNT: ", count)
        #     count = 0
        #     print(matrix)

        partial_info_binary_matrices += bot_partial_info_binary_matrices
        # print("debug_partial_info_binary_matrices: ", partial_info_binary_matrices)
        # count = 0
        # for matrix in input_partial_info_binary_matrices:
        #     # print("MATRIX: ", matrix)
        #     for matrix2 in matrix:
        #         # print("MATRIX2: ", matrix2)
        #         for row in matrix2:
        #             for col in row:
        #                 if col==1:
        #                     count+=1
        #         print("debug_partial_info_binary_matrices COUNT: ", count)
        #         count = 0
        #         print(matrix2)

        # print("images count: ", len(partial_info_binary_matrices))
        # for image in partial_info_binary_matrices:
        #     print("image: ", image)
            # print("image: ", image[0])
            # count = 0
            # for row in image[0]:
            #     for col in row:
            #         if col == 1:
            #             count += 1
            # print("count: ", count)

        action_binary_matrices += bot_action_binary_matrices
        scores += bot_scores

    # rollout data is generated after the normal data of each map..
    # ..because when splitting the data at training, if the the normal data is created and the then the rollout..
    # .., the training and validation sets will have the same maps

    if rollout:
        # print("Generating rollout data...")
        generate_data_rollout(path_matrices, comm_path_matrices, partial_info_binary_matrices,
                                action_binary_matrices, scores, total_steps, num_robots)

    # end = time.time()
    # time_taken = (end - start)/60
    # print("Iteration: {}, Planner: {}, Time taken: {:.3f}".format(i, planner, time_taken))

    print("path_matrices: ", len(path_matrices))
    print("partial_info_binary_matrices: ", len(partial_info_binary_matrices))
    print("actions_binary_matrices", len(action_binary_matrices))
    print("scores: ", len(scores))

def generate_data_rollout(path_matrices, comm_path_matrices, partial_info_binary_matrices, actions_binary_matrices, scores, total_steps, num_robots):
    temp_partial_info_binary_matrices = list()
    temp_path_matrices = list()
    temp_comm_path_matrices = list()
    temp_actions_binary_matrices = list()
    temp_scores = list()

    # "- ((steps*num_robots)+num_robots)" this is added because remember that rollout data is added..
    # to "input_partial_info_binary_matrices" so we need to start after the rollout data from the previous iteration
    index1 = len(partial_info_binary_matrices) - (total_steps*num_robots)

    curr_robot =  1
    # the +1 in "boundary_increment" is because there is initial position matrix added to the arrays
    boundary_increment = (total_steps * curr_robot)
    boundary = index1 + boundary_increment - 1
    
    # the while loop is to make sure we don't iterate through the entire dataset to create..
    # ..rollout data because we are matching current data with future data
    while index1 <= (len(partial_info_binary_matrices) - total_steps//2):
        if (curr_robot <= num_robots) and (index1 == boundary-(total_steps/5)):
            curr_robot += 1
            # index1 becomes the previous boundary
            index1 = boundary + 1
            # we move boundary forward by boundary increment
            boundary += boundary_increment
        
        temp_partial_info_binary_matrices.append(partial_info_binary_matrices[index1])

        index2 = randint(index1, boundary-2)
        # print("index2: ", index2)
        comm_path_index = randint(index1, index2)
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
        
        curr_path_matrix = path_matrices[index2]
        curr_comm_path_matrix = comm_path_matrices[comm_path_index]

        # iterate over each row, col idx of the np array and modify path_matrix
        for irow, icol, in np.ndindex(curr_comm_path_matrix.shape):
            if curr_comm_path_matrix[irow, icol] == 1:
                curr_path_matrix[irow, icol]=1

        temp_path_matrices.append(curr_path_matrix)
        # these are just fillers so we get the correct idx in the next iteration
        # temp_comm_path_matrices.append(curr_comm_path_matrix)
        temp_comm_path_matrices.append(1)

        temp_actions_binary_matrices.append(actions_binary_matrices[index2])
        temp_scores.append(scores[index2])

        index1 += 1

    partial_info_binary_matrices += temp_partial_info_binary_matrices
    path_matrices += temp_path_matrices
    comm_path_matrices += temp_comm_path_matrices
    actions_binary_matrices += temp_actions_binary_matrices
    scores += temp_scores

def generate_tensor_images(path_matricies, partial_info_binary_matrices, actions_binary_matrices, scores, outfile):
    data = list()

    for i in tqdm(range(len(partial_info_binary_matrices))):
        image = list()

        for partial_info in partial_info_binary_matrices[i]:
            image.append(partial_info)

        image.append(path_matricies[i])

        for action in actions_binary_matrices[i]:
            image.append(action)

        # this is needed, because torch complains otherwise that converting a list is too slow
        # it's better to use a np array because of the way a np array is stored in memory (contiguous)
        image = np.array(image)

        data.append([torch.IntTensor(image), scores[i]])

    # pickle progress
    print("Pickling started!")
    outfile_tensor_images = open(outfile, 'wb')
    pickle.dump(data, outfile_tensor_images)
    outfile_tensor_images.close()
    print("Pickling done!")


def main():

    #### SETUP ####

    BOUNDS = [21, 21]
    OCC_DENSITY = 18
    TRIALS = 1
    TOTAL_STEPS = 25
    NUM_ROBOTS = 3
    FULLCOMM_STEP = 1
    PARTIALCOMM_STEP = 5
    POORCOMM_STEP = 10

    CONF = get_CONF()
    json_comp_conf = get_json_comp_conf()
    neural_model = get_neural_model(CONF, json_comp_conf, BOUNDS)

    planner_options = [RandomPlanner(FULLCOMM_STEP, "full"), 
                       CellCountPlanner(neural_model[0], FULLCOMM_STEP, "full")]

    # for data generation
    generate_data = True
    rollout = True
    partial_info_binary_matrices = list()
    path_matrices = list()
    comm_path_matrices = list()
    actions_binary_matrices = list()
    scores = list()

    # for pickling data
    saved_scores = dict()

    datafile = "test"
    outfile_tensor_images = CONF[json_comp_conf]["pickle_path"] + datafile

    #### RUN ROBOTS ####

    for i in tqdm(range(TRIALS)):
        print("TRIAL: {}".format(i+1))

        ground_truth_map = GroundTruthMap(BOUNDS, OCC_DENSITY)
        belief_map = BeliefMap(BOUNDS)
        # robot_start_loc = get_random_loc(ground_truth_map)
        robot_start_loc = [get_random_loc(
            ground_truth_map) for _ in range(NUM_ROBOTS)]

        for planner in planner_options:
            saved_scores[planner.get_name()] = list()
            robots = get_robots(NUM_ROBOTS, belief_map,
                                ground_truth_map, robot_start_loc)

            robot_occupied_locs = set()  # so that we can calculate unique occupied cells observed for the score
            cum_score = 0
            for step in range(TOTAL_STEPS):
                robot_curr_locs = [bot.get_loc() for bot in robots]
                step_score = 0

                # run multiple robots in same map
                for bot in robots:
                    bot_simulator = bot.get_simulator()
                    bot_belief_map = bot.get_belief_map()

                    bot_simulator.run(planner, robot_curr_locs, robot_occupied_locs,
                                        robots, step, neural_model[0], device=neural_model[1])
                    robot_occupied_locs = robot_occupied_locs.union(bot_belief_map.get_occupied_locs())
                    step_score += bot_simulator.get_curr_score()
                    bot_simulator.reset_score() # needs to be reset otherwise the score will carry on to the next iteration

                    # bot_simulator.visualize(robots, step)

                cum_score += step_score

            # vis_map(robots, BOUNDS, belief_map)
            # vis_map(robots, BOUNDS, ground_truth_map)

            print("CUM_SCORE: ", cum_score)
            saved_scores[planner.get_name()].append(cum_score)

            if generate_data:
                print("Generating data matrices and rollout is {}..".format(rollout))
                generate_binary_matrices(robots, path_matrices, comm_path_matrices,
                                         partial_info_binary_matrices, actions_binary_matrices, scores, TOTAL_STEPS, NUM_ROBOTS, rollout)
                generate_tensor_images(path_matrices, partial_info_binary_matrices, actions_binary_matrices, 
                                       scores, outfile_tensor_images)
    
    plot_scores(saved_scores)
    

if __name__ == "__main__":

    main()