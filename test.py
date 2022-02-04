from zmq import device
from SensorModel import SensorModel
from Map import Map
from Robot import Robot
from Simulator import Simulator
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random as random
import time as time
import pickle
import NeuralNet
import torch
from tqdm import tqdm
from basic_MCTS_python.reward import reward_greedy

import cProfile
import pstats

# used to create random, valid starting locs
def get_random_loc(map, bounds):
    valid_starting_loc = False
    while not valid_starting_loc:
        x = random.randint(0, bounds[0]-1)
        y = random.randint(0, bounds[0]-1)
        valid_starting_loc = map.check_loc(x, y)

    return [x, y]

def oracle_visualize(robots, bounds, map, planner, reward_type=None, rollout_type=None):
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

        bot_color = bot.get_color()

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
    
    if reward_type and rollout_type is not None:
        plt.title(planner + "_" + reward_type + "_" + rollout_type)
    else:
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
        
        
if __name__ == "__main__":

    # Bounds need to be an odd number for the action to always be in the middle
    # greedy-o: greedy oracle (knows where the obstacles are in map)
    # greedy-no: greedy non-oracle (counts total unobserved cells in map)
    # planner_options = ["random", "greedy-o", "greedy-no", "net_nocomm", "net_everystep"]
    # planner_options = ["random", "greedy-o", "greedy-no", "net_everystep"]
    # planner_options = ["random", "greedy-o", "greedy-no", "net_everystep", "mcts"]
    # planner_options = ["random", "greedy-o_everyxstep", "greedy-o", "greedy-no_everyxstep", "greedy-no", "net_everyxstep", "net_everystep", "mcts"]
    # planner_options = ["random", "greedy-o", "greedy-no", "mcts"]
    # planner_options = ["random", "greedy-o", "greedy-no"]
    planner_options = ["mcts"]
    # network_options = ["net_nocomm", "net_everyxstep", "net_everystep"]
    # network_options = ["net_everystep"]
    # rollout_options = ["random", "greedy"]
    # rollout_options = ["random"]
    # rollout_options = ["random", "greedy", "net_everyxstep", "net_everystep"]
    rollout_options = ["random"]
    # rollout_options = ["random", "greedy"] + network_options
    # reward_options = ["greedy", "net_everyxstep", "net_everystep"]
    reward_options = ["random"]
    # reward_options = network_options
    # reward_options = ["random"]
    # reward_options = ["random", "greedy"]
    # reward_options = ["greedy"]
    print("reward convergence test")
    bounds = [21, 21]
    trials = 1
    steps = 25
    num_robots = 1
    # to decide which step the bot communicates
    comm_step = 3
    # obs_occupied_oracle = set() # this is for calculating the end score counting only unique seen cells
    visualize = False
    # profiling functions
    profile = False

    if profile:
        pr = cProfile.Profile()
        pr.enable()

    # this is for pickling and visualizing the data -> check pickle_script.py
    if "mcts" in planner_options:
        # score_lists = [list() for _ in range((len(planner_options)-1)+(len(rollout_options)*len(reward_options)))]
        score_lists = [list() for _ in range((len(planner_options)-1)+(len(rollout_options)*len(reward_options)))]
    else:
        score_lists = [list() for _ in range(len(planner_options))]
    
    # load neural net
    # weight_file = "circles_21x21_epoch3_random_greedyo_r4_t1000_s50_norollout_diffstartloc"
    weight_file = "circles_21x21_epoch1_random_greedyo_r4_t2000_s25_rollout_diffstartloc"
    # weight_file = "circles_21x21_epoch1_random_greedyo_r4_t2000_s25_rollout_diffstartloc_iter2"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)
    neural_model = NeuralNet.Net(bounds).to(device)
    # alienware
    neural_model.load_state_dict(torch.load("/home/kavi/thesis/neural_net_weights/"+weight_file)) 
    # macbook 
    # neural_model.load_state_dict(torch.load("/Users/kavisen/osu_thesis/"+weight_file)) 
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # neural_model.to(device) 
    neural_model.eval()

    # this is for pickling the score_lists
    # alienware
    # filename = '/home/kavi/thesis/pickles/planner_scores_multibot/trial{}_steps{}_roll_random_greedy_rew_net_everystep_wbacktracking'.format(trials, steps)
    filename = '/home/kavi/thesis/pickles/planner_scores_multibot/test2'
    # filename = '/home/kavi/thesis/pickles/planner_scores_multibot/trial{}_steps{}_roll_random_greedy_net_everystep_rew_greedy_net_everystep_notimesvisited'.format(trials, steps)
    # filename = '/home/kavi/thesis/pickles/planner_scores_multibot/trial{}_steps{}_comm_nocomm_commstep3'.format(trials, steps)
    # macbook
    # filename = '/Users/kavisen/osu_thesis/pickles/planner_scores_test'

    debug_mcts_reward_greedy_list = list()
    debug_mcts_reward_network_list = list()

    test_start_time = time.time()
    for i in tqdm(range(trials)):
        trial_start_time = time.time()
        print("TRIAL NO: {}".format(i+1))
        map = Map(bounds, 7, (), False)
        # unobs_occupied = copy.deepcopy(map.get_unobs_occupied())
        unobs_occupied = map.get_unobs_occupied()
        bots_starting_locs = list()
        
        # for pickling data
        score_list = 0

        # create robots
        robots = list()
        # give all robots the same start loc to force communication for testing
        start_loc = get_random_loc(map, bounds)
        for _ in range(num_robots):
            # start_loc = get_random_loc(map, bounds)
            bot = Robot(start_loc[0], start_loc[1], bounds, map)
            robots.append(bot)
            bots_starting_locs.append(start_loc)

        for planner in planner_options:
            print("Planner: {}".format(planner))

            if planner == "mcts":
                for rollout_type in rollout_options:
                    for reward_type in reward_options:
                        if rollout_type == "net_everystep" and reward_type == "net_everyxstep":
                            continue
                        if rollout_type == "net_everyxstep" and reward_type == "net_everystep":
                            continue
                        print("Rollout: {}, Reward: {}".format(rollout_type, reward_type))

                        # this is for pickling the data
                        curr_list = score_lists[score_list]
                        if len(curr_list) == 0:
                            curr_list.append(rollout_type + '_' + reward_type)
                        score_list += 1

                        obs_occupied_oracle = set() # this is for calculating the end score counting only unique seen cells
                        obs_free_oracle = set()

                        # the map has to be the same for each planner
                        for bot in robots:
                            map = Map(bounds, 7, copy.deepcopy(unobs_occupied), True)
                            sensor_model = SensorModel(bot, map)
                            simulator = Simulator(map, bot, sensor_model, planner, rollout_type, reward_type)
                            # start_loc = bot.get_start_loc()
                            # bot.set_loc(start_loc[0], start_loc[1])
                            bot.set_loc(6, 7)
                            bot.add_map(map)
                            bot.add_sensor_model(sensor_model)
                            bot.add_simulator(simulator)
                            # this adds the initial matrices to appropriate lists
                            bot_simulator = bot.get_simulator()
                            bot_simulator.initialize_data(bots_starting_locs, obs_occupied_oracle)
                            # this is needed incase any locations are scanned in the initial position
                            obs_occupied_oracle = obs_occupied_oracle.union(bot_simulator.get_obs_occupied())
                            obs_free_oracle = obs_free_oracle.union(bot_simulator.get_obs_free())
                            
                        steps_count = 0
                        for step in tqdm(range(steps)):
                            curr_robot_positions = set()
                        
                            for bot in tqdm(robots):
                                simulator = bot.get_simulator()
                                sensor_model = bot.get_sensor_model()

                                # oracle_visualize(robots, bounds, map, planner, reward_type, rollout_type)

                                simulator.run(neural_model, curr_robot_positions, train=True, debug_mcts_reward_greedy_list=debug_mcts_reward_greedy_list, debug_mcts_reward_network_list=debug_mcts_reward_network_list, device=device)

                                # communicate(robots, obs_occupied_oracle, obs_free_oracle)

                                # to keep track of score
                                obs_occupied_oracle = obs_occupied_oracle.union(simulator.get_obs_occupied())
                                obs_free_oracle = obs_free_oracle.union(bot_simulator.get_obs_free())

                            steps_count += 1
                            if rollout_type == "net_everystep" or reward_type == "net_everystep":   
                                communicate(robots, obs_occupied_oracle, obs_free_oracle)
                            elif rollout_type == "net_everyxstep" and steps_count%comm_step==0:   
                                communicate(robots, obs_occupied_oracle, obs_free_oracle)
                            elif reward_type == "net_everyxstep" and steps_count%comm_step==0:   
                                communicate(robots, obs_occupied_oracle, obs_free_oracle)
                            elif rollout_type == "net_nocomm" or reward_type == "net_nocomm":
                                pass
                            elif rollout_type == "greedy" and reward_type == "greedy":
                                communicate(robots, obs_occupied_oracle, obs_free_oracle)
                            elif rollout_type == "random" and reward_type == "greedy":
                                communicate(robots, obs_occupied_oracle, obs_free_oracle)
                            steps_end = time.time()

                        score = len(obs_occupied_oracle)     
                        print("Score: ", score)
                        curr_list.append(score)

                        oracle_visualize(robots, bounds, map, planner, reward_type, rollout_type)

                        # pickle progress
                        outfile = open(filename,'wb')
                        pickle.dump(score_lists, outfile)
                        outfile.close()


            else: # these are the myopic planners

                # adds planner name to the visualization list
                curr_list = score_lists[score_list]
                if len(curr_list) == 0:
                    curr_list.append(planner)
                score_list += 1

                obs_occupied_oracle = set() # this is for calculating the end score counting only unique seen cells
                obs_free_oracle = set()
            
                # the map has to be the same for each planner
                for bot in robots:
                    map = Map(bounds, 7, copy.deepcopy(unobs_occupied), True)
                    sensor_model = SensorModel(bot, map)
                    simulator = Simulator(map, bot, sensor_model, planner)
                    # start_loc = bot.get_start_loc()
                    bot.set_loc(start_loc[0], start_loc[1])
                    bot.add_map(map)
                    bot.add_sensor_model(sensor_model)
                    bot.add_simulator(simulator)
                    # this adds the initial matrices to appropriate lists
                    bot_simulator = bot.get_simulator()
                    bot_simulator.initialize_data(bots_starting_locs, obs_occupied_oracle)
                    # this is needed incase any locations are scanned in the initial position
                    obs_occupied_oracle = obs_occupied_oracle.union(bot_simulator.get_obs_occupied())
                    # obs_free_oracle = obs_free_oracle.union(bot_simulator.get_obs_free())
                    obs_free_oracle = obs_free_oracle.union(simulator.get_obs_free())

                steps_count = 0
                for step in range(steps):
                    curr_robot_positions = set()

                    # run multiple robots in same map
                    for bot in robots:
                        simulator = bot.get_simulator()
                        sensor_model = bot.get_sensor_model()

                        if visualize:
                            simulator.visualize()

                        # we run it without obs_occupied_oracle because if not the normal planners have oracle info
                        simulator.run(neural_model, curr_robot_positions, train=True, device=device)

                        # to keep track of score
                        obs_occupied_oracle = obs_occupied_oracle.union(simulator.get_obs_occupied())
                        # obs_free_oracle = obs_free_oracle.union(bot_simulator.get_obs_free())
                        obs_free_oracle = obs_free_oracle.union(simulator.get_obs_free())

                        # if planner == "network":
                            # communicate(robots, obs_occupied_oracle, obs_free_oracle)

                    steps_count += 1
                    if planner == "net_everystep":    
                        communicate(robots, obs_occupied_oracle, obs_free_oracle)
                    elif planner == "greedy-o_everyxstep" and steps_count%comm_step==0:   
                        communicate(robots, obs_occupied_oracle, obs_free_oracle)
                    elif planner == "greedy-no_everyxstep" and steps_count%comm_step==0:   
                        communicate(robots, obs_occupied_oracle, obs_free_oracle)    
                    elif planner == "net_everyxstep" and steps_count%comm_step==0:   
                        communicate(robots, obs_occupied_oracle, obs_free_oracle)
                    elif planner == "net_nocomm":
                        pass
                    elif planner == "greedy-o" or planner == "greedy-no":
                        communicate(robots, obs_occupied_oracle, obs_free_oracle)
                    steps_end = time.time()

                score = len(obs_occupied_oracle)     
                print("Score: ", score)
                curr_list.append(score)
                oracle_visualize(robots, bounds, map, planner)

                # if planner == "net_nocomm" or planner == "net_everyxstep" or planner == "net_everystep":
                    # oracle_visualize(robots, bounds, map, planner)

                # pickle progress
                outfile = open(filename,'wb')
                pickle.dump(score_lists, outfile)
                outfile.close()

        trial_end_time = time.time()
        print("Trial time taken (mins): ", (trial_end_time - trial_start_time)/60)
    
    # profiling performance of code
    if profile:
        pr.disable()
        pr.print_stats()
        with open("cProfile_stats_multirobot2.txt", "w") as f:
            ps = pstats.Stats(pr, stream=f)
            ps.sort_stats('cumtime')
            ps.print_stats()

    # create bar graphs
    bars = list()
    scores = list()

    for score_list in score_lists:
        if len(score_list) == 0: # this condition was added because we are skipping some mcts planners
            continue
        planner_name = score_list[0]
        bars.append(planner_name)
        del score_list[0]
        curr_score = sum(score_list)/len(score_list)
        scores.append(curr_score)

    x_pos = np.arange(len(bars))
    # bar chart plot
    # plt.bar(x_pos, scores, color=['#33e6ff', 'red', 'green', 'blue', '#FFC0CB', '#800080', '#fdbe83', '#00ab66', '#0b1320', '#ddceff', '#4000ff', '#ff876f', '#540077'])
    # plt.xticks(x_pos, bars, rotation=45)
    # plt.title(weight_file)
    
    # # puts the value on top of each bar
    # for i in range(len(bars)):
    #     plt.text(i, scores[i], scores[i], ha = 'center')

    # plt.show()

    # box plot
    score_lists_copy = score_lists
    
    for score_list in score_lists_copy:
        if len(score_list) == 0: # this condition was added because we are skipping some mcts planners
            continue
        score_list.remove(score_list[0])   

    # do this otherwise x axis is not correct
    for i in x_pos:
        x_pos[i] += 1

    fig = plt.figure(figsize =(10, 7))
    # Creating axes instance
    plt.boxplot(score_lists_copy)
    plt.xticks(x_pos, bars, rotation=25)
    plt.title(weight_file+"_trials:"+str(trials)+"_steps:"+str(steps))
    plt.show()



