import matplotlib.pyplot as plt
import matplotlib.patches as patches
from GroundTruthMap import GroundTruthMap
from utils import get_random_loc, communicate, get_CONF, get_json_comp_conf
import torch
from NeuralNet import Net
from time import time
from tqdm import tqdm
from BeliefMap import BeliefMap
from Robot import Robot
from SensorModel import SensorModel
from Simulator import Simulator
from copy import deepcopy

def vis_belief_map(robots, bounds, belief_map):
    plt.xlim(0, bounds[0])
    plt.ylim(0, bounds[1])

    ax = plt.gca()
    ax.set_aspect('equal', 'box')

    # this has to be done before the bot for loop to avoid red patches
    # ..going over the other obs_occupied patches
    for spot in belief_map.get_unknown_locs():
    # for spot in map.unobs_occupied:
    # for spot in map.unobs_free:
        hole = patches.Rectangle(spot, 1, 1, facecolor='black')
        ax.add_patch(hole)

    # get all the observed locations from all robots
    free_locs = set()
    occupied_locs = set()
    for bot in robots:
        # bot_map = bot.get_map()
        bot_map = bot.get_belief_map()

        # obs_free = obs_free.union(bot_map.get_obs_free())
        free_locs = free_locs.union(bot_map.get_free_locs())
        # obs_occupied = obs_occupied.union(bot_map.get_obs_occupied())
        occupied_locs = obs_occupied.union(bot_map.get_occupied_locs())

        bot_color = bot.get_color()

        # plot robot
        robot_x = bot.get_loc()[0] + 0.5
        robot_y = bot.get_loc()[1] + 0.5
        plt.scatter(robot_x, robot_y, color=bot_color, zorder=5)

        # plot robot path
        x_values = list()
        y_values = list()
        # for path in bot.get_sensor_model().get_final_path():
        for path in bot.get_exec_path():
            x_values.append(path[0] + 0.5)
            y_values.append(path[1] + 0.5)
        plt.plot(x_values, y_values, color=bot_color)

        # this is in the loop so that we can use diff colors for each robot's occ cells 
        for spot in occupied_locs:
            hole = patches.Rectangle(spot, 1, 1, facecolor=bot_color)
            ax.add_patch(hole)
        obs_occupied = set()

    for spot in free_locs:
        hole = patches.Rectangle(spot, 1, 1, facecolor='white')
        ax.add_patch(hole)

    plt.show()

def get_neural_model(CONF, bounds):
    weight_file = "circles_21x21_epoch1_random_greedyno_r4_t2000_s35_rolloutotherpath_samestartloc_obsdensity18"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)
    neural_model = Net(bounds).to(device)
    neural_model.load_state_dict(torch.load(
        CONF[json_comp_conf]["neural_net_weights_path"]+weight_file))
    neural_model.eval()
    
    return neural_model, device

def get_robots(num_robots, belief_map, ground_truth_map, planner, robots_start_locs):
    robots = set()
    start_loc = get_random_loc(ground_truth_map)
    for _ in range(num_robots):
        bot = Robot(start_loc[0], start_loc[1], deepcopy(belief_map))
        sensor_model = SensorModel(bot, belief_map)
        simulator = Simulator(belief_map, bot, sensor_model, planner, generate_data=False)
        # start_loc = get_random_loc(belief_map)
        bot.set_sensor_model(sensor_model)
        bot.set_simulator(simulator)
        robots.add(bot)

    return robots


if __name__ == "__main__":

    CONF = get_CONF()
    json_comp_conf = get_json_comp_conf()

    BOUNDS = [21, 21]
    TRIALS = 100
    TOTAL_STEPS = 25
    NUM_ROBOTS = 3
    neural_model = get_neural_model(CONF, BOUNDS)
    planner_options = ["random_fullcomm"]

    score_occ_locs = set()
    for i in tqdm(range(TRIALS)):
        trial_start = time() 
        print("TRIAL: {}".format(i+1))
        
        OCC_DENSITY = 18
        ground_truth_map = GroundTruthMap(BOUNDS, OCC_DENSITY)
        belief_map = BeliefMap(BOUNDS)
        robot_start_locs = list()

        for planner in planner_options:

            robots = get_robots(NUM_ROBOTS, belief_map, ground_truth_map, planner, robot_start_locs)

            # this is for calculating the end score counting only unique seen cells
            obs_occupied_global = set()
            obs_free_global = set()

            for bot in robots:
                bot_simulator = bot.get_simulator()
                bot_simulator.initialize_data(robot_start_locs)

            for step in range(TOTAL_STEPS):
                robot_curr_locs = set()

                # run multiple robots in same map
                for bot in robots:
                    bot_simulator = bot.get_simulator()
                    bot_belief_map = bot.get_belief_map()
                    bot_sensor_model = bot.get_sensor_model()

                    bot_simulator.run(neural_model[0], robot_curr_locs, device=neural_model[1])

                    # to keep track of score
                    score_occ_locs = score_occ_locs.union(bot_belief_map.get_obs_occupied())
                
                step_score = len(score_occ_locs)

                # COMMUICATE FUNCTION
                steps_end = time.time()

            trial_score = len(score_occ_locs)
            print("Score: ", trial_score)
            # curr_list.append(score)
            # oracle_visualize(robots, BOUNDS, map, planner)

            # pickle progress
            # outfile = open(filename, 'wb')
            # pickle.dump(score_lists, outfile)
            # outfile.close()

            # accsore_list.append(acc_step_score)
            # outfile = open(filename_accscore, 'wb')
            # pickle.dump(accsore_list, outfile)
            # outfile.close()
