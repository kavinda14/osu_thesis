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

def vis_map(robots, bounds, belief_map=None, ground_truth_map=None):
    plt.xlim(0, bounds[0])
    plt.ylim(0, bounds[1])

    ax = plt.gca()
    ax.set_aspect('equal', 'box')

    # this has to be done before the bot for loop to avoid red patches
    # ..going over the other obs_occupied patches
    if ground_truth_map is None:
        for spot in belief_map.get_unknown_locs():
            hole = patches.Rectangle(spot, 1, 1, facecolor='black')
            ax.add_patch(hole)
    else:
        # color all occupied locs before putting specific bot colors on them (to identify which bot discovered what)
        occupied_locs = ground_truth_map.get_occupied_locs()
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

        # this is in the loop so that we can use diff colors for each robot's occ cells 
        for spot in occupied_locs:
            hole = patches.Rectangle(spot, 1, 1, facecolor=bot_color)
            ax.add_patch(hole)

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

def get_robots(num_robots, belief_map, ground_truth_map, robot_start_loc):
    robots = set()
    for _ in range(num_robots):
        belief_map_copy = deepcopy(belief_map) # to make sure that each robot has a diff belief map object
        bot = Robot(robot_start_loc[0], robot_start_loc[1], belief_map_copy)
        sensor_model = SensorModel(bot, belief_map)
        simulator = Simulator(belief_map_copy, ground_truth_map, bot, sensor_model, generate_data=False)
        bot.set_sensor_model(sensor_model)
        bot.set_simulator(simulator)
        robots.add(bot)

    return robots


if __name__ == "__main__":

    CONF = get_CONF()
    json_comp_conf = get_json_comp_conf()

    BOUNDS = [21, 21]
    OCC_DENSITY = 18
    TRIALS = 1
    TOTAL_STEPS = 25
    NUM_ROBOTS = 3
    ACTIONS = ['left', 'right', 'backward', 'forward']
    FULLCOMM_STEP = 1
    PARTIALCOMM_STEP = 5
    POORCOMM_STEP = 10
    neural_model = get_neural_model(CONF, BOUNDS)

    score_occ_locs = set()
    for i in tqdm(range(TRIALS)):
        trial_start = time() 
        print("TRIAL: {}".format(i+1))
        
        ground_truth_map = GroundTruthMap(BOUNDS, OCC_DENSITY)
        belief_map = BeliefMap(BOUNDS)
        robot_start_loc = get_random_loc(ground_truth_map)
        # deepcopy the map because we need the same map in the trial for each planner
        planner_options = [RandomPlanner(ACTIONS, FULLCOMM_STEP), CellCountPlanner(ACTIONS, neural_model[0], FULLCOMM_STEP)]

        for planner in planner_options:
            robots = get_robots(NUM_ROBOTS, belief_map, ground_truth_map, robot_start_loc)
            # initialize matrices for network
            for bot in robots:
                bot_simulator = bot.get_simulator()
                bot_simulator.initialize_data(robot_start_loc)
            
            robot_occupied_locs = set() # so that we can calculate unique occupied cells observed for the score
            
            cum_score = 0
            for step in range(TOTAL_STEPS):
                robot_curr_locs = list()
                step_score = 0

                for bot in robots:
                    robot_curr_locs.append(bot.get_loc())

                # run multiple robots in same map
                for bot in robots:
                    bot_simulator = bot.get_simulator()
                    bot_belief_map = bot.get_belief_map()
                    bot_sensor_model = bot.get_sensor_model()

                    bot_simulator.run(planner, robot_curr_locs, robot_occupied_locs, robots, step, neural_model[0], device=neural_model[1])
                    robot_occupied_locs = robot_occupied_locs.union(bot_belief_map.get_occupied_locs())

                    # bot_simulator.visualize(robots, step)

                    step_score += bot_simulator.get_score()
                    bot_simulator.reset_score() # needs to be reset otherwise the score will carry on to the next iteration
                
                # vis_map(robots, BOUNDS, belief_map=belief_map)
                # vis_map(robots, BOUNDS, ground_truth_map=ground_truth_map)

                cum_score += step_score
            
            steps_end = time()
            vis_map(robots, BOUNDS, belief_map=belief_map)
            vis_map(robots, BOUNDS, ground_truth_map=ground_truth_map)

            print("CUM_SCORE: ", cum_score)

            # curr_list.append(score)

            # pickle progress
            # outfile = open(filename, 'wb')
            # pickle.dump(score_lists, outfile)
            # outfile.close()

            # accsore_list.append(acc_step_score)
            # outfile = open(filename_accscore, 'wb')
            # pickle.dump(accsore_list, outfile)
            # outfile.close()
