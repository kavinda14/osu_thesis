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

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# used to create random, valid starting locs
def get_random_loc(map, bounds):
    valid_starting_loc = False
    while not valid_starting_loc:
        x = random.randint(0, bounds[0]-1)
        y = random.randint(0, bounds[0]-1)
        valid_starting_loc = map.check_loc(x, y)

    return [x, y]

def oracle_visualize(robots, bounds, map, planner):
    plt.xlim(0, bounds[0])
    plt.ylim(0, bounds[1])
    # plt.title("Planner: {}, Score: {}".format(self.planner, sum(self.sensor_model.get_final_scores())))

    ax = plt.gca()
    ax.set_aspect('equal', 'box')

    # this has to be done before the bot for loop to avoid red patches
    # ..going over the other obs_occupied patches
    for spot in map.unobs_occupied:
        hole = patches.Rectangle(spot, 1, 1, facecolor='red')
        ax.add_patch(hole)

    for spot in map.unobs_free:
        hole = patches.Rectangle(spot, 1, 1, facecolor='black')
        ax.add_patch(hole)
    
    # get all the observed locations from all robots
    obs_free = set()
    obs_occupied = set()
    for bot in robots:
        simulator = bot.get_simulator()
        map = bot.get_map()

        obs_free = obs_free.union(simulator.get_obs_free())
        obs_occupied = obs_occupied.union(simulator.get_obs_occupied())

        # add unique color for robot
        r = random.random()
        b = random.random()
        g = random.random()
        bot_color = (r, g, b)

        # plot robot
        robot_x = bot.get_loc()[0] + 0.5
        robot_y = bot.get_loc()[1] + 0.5
        plt.scatter(robot_x, robot_y, color=bot_color, zorder=5)

        # plot robot path
        x_values = list()
        y_values = list()
        for path in bot.get_sensor_model().get_final_path():
            x_values.append(path[0] + 0.5)
            y_values.append(path[1] + 0.5)
        plt.plot(x_values, y_values, color=bot_color)

        for spot in obs_occupied:
            hole = patches.Rectangle(spot, 1, 1, facecolor=bot_color)
            ax.add_patch(hole)
        obs_occupied = set()
    
    for spot in obs_free:
        hole = patches.Rectangle(spot, 1, 1, facecolor='white')
        ax.add_patch(hole)

    plt.title(planner)

    plt.show()

def communicate(robots, obs_occupied_oracle, obs_free_oracle):
    for bot1 in robots:
        sensor_model_bot1 = bot1.get_sensor_model()
        map_bot1 = bot1.get_map()
        other_paths = list()

        # for communicating the belief maps
        # by communicating these sets, the maps will contain these updates
        map_bot1.add_oracle_obs_free(obs_free_oracle)
        map_bot1.add_oracle_obs_occupied(obs_occupied_oracle)

        for bot2 in robots:
            if bot1 is not bot2:
                sensor_model_bot2 = bot2.get_sensor_model()
                final_path_bot2 = sensor_model_bot2.get_final_path()
                other_paths += final_path_bot2
                
        final_other_path_bot1 = sensor_model_bot1.get_final_other_path() + other_paths
        # final_other_path_bot1 = sensor_model_bot1.get_final_other_path().union(other_paths)
        sensor_model_bot1.set_final_other_path(final_other_path_bot1)

# rollout produces the unique data needed for mcts rollout
# this is done because in rollout, belief map stays the same even though path and actions change


def generate_data_matrices(trials, steps, num_robots, planner_options, visualize, bounds, outfile, rollout=True):
    input_partial_info_binary_matrices = list()
    input_path_matrices = list()
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
        # start_loc = get_random_loc(map, bounds)
        for _ in range(num_robots):
            start_loc = get_random_loc(map, bounds)
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
                bot_simulator.initialize_data(bots_starting_locs, obs_occupied_oracle)
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
                    simulator.run(False, curr_robot_positions=set(), obs_occupied_oracle=obs_occupied_oracle, train=False)
                    # print("DEBUG PARTIAL IMAGE: ", sensor_model.get_final_partial_info()[-1])
                    # print("DEBUG SCORES: ", sensor_model.get_final_scores())

                    obs_occupied_oracle = obs_occupied_oracle.union(simulator.get_obs_occupied())
                    obs_free_oracle = obs_free_oracle.union(bot_simulator.get_obs_free())
                    
                    communicate(robots, obs_occupied_oracle, obs_free_oracle)

                    if visualize: 
                        simulator.visualize() 

                # communicate(robots, obs_occupied_oracle, obs_free_oracle)
            
            # oracle_visualize(robots, bounds, map, planner)

            ### DATA MATRICES
            for bot in robots:
                sensor_model = bot.get_sensor_model()

                path_matricies = sensor_model.get_final_path_matrices()
                         
                final_partial_info = sensor_model.get_final_partial_info()
                partial_info_binary_matrices = sensor_model.create_binary_matrices(final_partial_info)

                final_actions = sensor_model.get_final_actions()
                final_actions_binary_matrices = sensor_model.create_binary_matrices(final_actions)

                final_scores = sensor_model.get_final_scores()

                input_path_matrices += path_matricies
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
            
            print("data length: ", len(input_path_matrices))
            
            if rollout:
                print("Generating rollout data...")
                generate_data_rollout(input_path_matrices, input_partial_info_binary_matrices, input_actions_binary_matrices, input_scores, steps, num_robots, outfile)        
        
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


def generate_data_rollout(input_path_matrices, input_partial_info_binary_matrices, input_actions_binary_matrices, input_scores, steps, num_robots, outfile):
    temp_input_partial_info_binary_matrices = list()
    temp_input_path_matrices = list()
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
        # debug
        # print()
        # print("index1: ", index1)
        # print("index2: ", index2)
        # print("boundary: ", boundary)
        # print()
        
        temp_input_path_matrices.append(input_path_matrices[index2])
        temp_input_actions_binary_matrices.append(input_actions_binary_matrices[index2])
        temp_input_scores.append(input_scores[index2])

        index1 += 1

    input_partial_info_binary_matrices += temp_input_partial_info_binary_matrices
    input_path_matrices += temp_input_path_matrices
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
        
        data.append([torch.IntTensor(image), final_scores[i]])

    # pickle progress
    print("Pickling started!")
    outfile_tensor_images = open(outfile, 'wb')
    pickle.dump(data, outfile_tensor_images)
    outfile_tensor_images.close()
    print("Pickling done!")


if __name__ == "__main__":

    # for pickling
    # alienware
    # outfile_tensor_images = '/home/kavi/thesis/pickles/data_21x21_circles_random_greedyo_r4_t1000_s50_norollout_diffstartloc'
    outfile_tensor_images = '/home/kavi/thesis/pickles/data_21x21_circles_random_greedyno_r4_t2000_s25_rollout_diffstartloc'
    # macbook
    # outfile_tensor_images = '/Users/kavisen/osu_thesis/data/data_21x21_circles_random_greedyno_r4_t800_s50_rollout'
    
    # generate data
    print("Generating matrices")
    planner_options = ["random", "greedy-no"]
    generate_data_matrices(trials=2000, steps=25, num_robots=4, planner_options=planner_options, visualize=False, bounds=[21, 21], outfile=outfile_tensor_images)
    