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

def oracle_visualize(robots, bounds, map):
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
        print(obs_occupied)
        obs_occupied = set()
    
    for spot in obs_free:
        hole = patches.Rectangle(spot, 1, 1, facecolor='white')
        ax.add_patch(hole)

    plt.show()

def communicate(robots):
    for bot1 in robots:
        sensor_model_bot1 = bot1.get_sensor_model()
        final_path_bot1 = sensor_model_bot1.get_final_path()
        # print("final_path_bot1", final_path_bot1)
        for bot2 in robots:
            if bot1 is not bot2:
                sensor_model_bot2 = bot2.get_sensor_model()
                final_other_path_bot2 = sensor_model_bot2.get_final_other_path() + final_path_bot1          
                sensor_model_bot2.set_final_other_path(final_other_path_bot2)

                # print("final_path_bot2", final_path_bot2)
                # print("final_other_path_bot2", final_other_path_bot2)
        # print()

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
        unobs_occupied = copy.deepcopy(map.get_unobs_occupied())

        # create robots
        robots = list()
        for _ in range(num_robots):
            start_loc = get_random_loc(map, bounds)
            bot = Robot(start_loc[0], start_loc[1], bounds, map)
            robots.append(bot)

        for planner in planner_options: 
            start = time.time()
            # Bounds need to be an odd number for the action to always be in the middle
            for bot in robots:
                map = Map(bounds, 7, copy.deepcopy(unobs_occupied), True)
                sensor_model = SensorModel(bot, map)
                simulator = Simulator(map, bot, sensor_model, planner)
                bot.add_map(map)
                bot.add_sensor_model(sensor_model)
                bot.add_simulator(simulator)

            for step in range(steps):

                # run multiple robots in same map
                for bot in robots:
                    simulator = bot.get_simulator()

                    if visualize:
                        simulator.visualize()

                    simulator.run(False)
                    
                    if visualize: 
                        simulator.visualize()

            ### TRAINING DATA    
            for bot in robots:
                sensor_model = bot.get_sensor_model()

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

                # generate_data_rollout(input_path_matrices, input_partial_info_binary_matrices, input_actions_binary_matrices, input_scores, steps, outfile)        
                    
                # end = time.time()
                # time_taken = (end - start)/60
                # print("Iteration: {}, Planner: {}, Time taken: {:.3f}".format(i, planner, time_taken))

                print("final_path_matrices: ", len(input_path_matrices))
                print("final_partial_info_binary_matrices: ", len(input_partial_info_binary_matrices))
                print("final_final_actions_binary_matrices", len(input_actions_binary_matrices))
                print("final_final_scores: ", len(input_scores))

                communicate(robots)

        oracle_visualize(robots, bounds, map)
                  
    print("final_path_matrices: ", len(input_path_matrices))
    print("final_partial_info_binary_matrices: ", len(input_partial_info_binary_matrices))
    print("final_final_actions_binary_matrices", len(input_actions_binary_matrices))
    print("final_final_scores: ", len(input_scores))

    generate_tensor_images(input_path_matrices, input_partial_info_binary_matrices, input_actions_binary_matrices, input_scores, outfile)


def generate_data_rollout(input_path_matrices, input_partial_info_binary_matrices, input_actions_binary_matrices, input_scores, steps, outfile):
    temp_input_partial_info_binary_matrices = list()
    temp_input_path_matrices = list()
    temp_input_actions_binary_matrices = list()
    temp_input_scores = list()

    # integer divide by two because we don't want to double the dataset size, but just a decent amount of samples
    # -5 because index 2 has to choose values ahead of index1
    # boundary = steps+1

    index1 = len(input_partial_info_binary_matrices) - steps
    boundary = index1 + steps

    while index1 < (len(input_partial_info_binary_matrices) - (steps//4)):
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
    # outfile_tensor_images = '/home/kavi/thesis/pickles/data_21x21_circles_random_greedyno_t800_s200_rollout'
    # macbook
    outfile_tensor_images = '/Users/kavisen/osu_thesis/test'
    
    # generate data
    print("Generating matrices")
    # planner_options = ["random", "greedy-o", "greedy-no"]
    # planner_options = ["random", "greedy-no"]
    # planner_options = ["random", "greedy-no"]
    planner_options = ["random"]
    generate_data_matrices(trials=3, steps=2, num_robots=2, planner_options=planner_options, visualize=False, bounds=[21, 21], outfile=outfile_tensor_images, rollout=False)
    