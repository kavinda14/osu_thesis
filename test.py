from SensorModel import SensorModel
from Map import Map
from Robot import Robot
from Simulator import Simulator
import copy
import matplotlib.pyplot as plt
import numpy as np
import time as time
import pickle
import NeuralNet
import torch
from tqdm import tqdm
from util import get_random_loc, oracle_visualize, communicate, get_CONF, get_json_comp_conf

import cProfile
import pstats


if __name__ == "__main__":

    CONF = get_CONF()
    json_comp_conf = get_json_comp_conf()

    # Bounds need to be an odd number for the action to always be in the middle
    # greedy-o: greedy oracle (knows where the obstacles are in map)
    # greedy-no: greedy non-oracle (counts total unobserved cells in map)
    # planner_options = ["random_poorcomm", "random_partialcomm", "random_fullcomm",
    #                    "greedy_poorcomm", "greedy_partialcomm", "greedy_fullcomm",
    #                    "net_poorcomm", "net_partialcomm", "net_fullcomm",
    #                    "mcts"]
    # planner_options = ["random_poorcomm", "random_partialcomm", "random_fullcomm",
    #                    "greedy_poorcomm", "greedy_partialcomm", "greedy_fullcomm",
    #                    "net_poorcomm", "net_partialcomm", "net_fullcomm"]
    planner_options = ["mcts"]
    # rollout_options = ["random_poorcomm", "random_partialcomm", "random_fullcomm",
    #                    "greedy_poorcomm", "greedy_partialcomm", "greedy_fullcomm",
    #                    "net_poorcomm", "net_partialcomm", "net_fullcomm"]
    rollout_options = ["random_fullcomm"]
    reward_options = ["net_fullcomm"]
    # reward_options = ["random"]
    # reward_options = ["greedy_poorcomm", "greedy_partialcomm", "greedy_fullcomm",
    #                   "net_poorcomm", "net_partialcomm", "net_fullcomm"]
    bounds = [21, 21]
    trials = 100
    steps = 25
    num_robots = 3
    # to decide which step the bot communicates
    partial_comm_step = 5
    poor_comm_step = 10
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
        score_lists = [list() for _ in range(
            (len(planner_options)-1)+(len(rollout_options)*len(reward_options)))]
    else:
        score_lists = [list() for _ in range(len(planner_options))]

    # load neural net
    # weight_file = "circles_21x21_epoch3_random_greedyo_r4_t1000_s50_norollout_diffstartloc"
    # weight_file = "circles_21x21_epoch1_random_greedyo_r4_t2000_s25_rollout_diffstartloc"
    weight_file = "circles_21x21_epoch1_random_greedyno_r4_t4000_s25_rolloutotherpath_samestartloc_commscorrected"
    # weight_file = "circles_21x21_epoch1_random_greedyno_r4_t2000_s25_rollout_diffstartloc_otherpathmix"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)
    neural_model = NeuralNet.Net(bounds).to(device)
    neural_model.load_state_dict(torch.load(
        CONF[json_comp_conf]["neural_net_weights_path"]+weight_file))
    neural_model.eval()

    # if testing other neural models
    # neural_model_trial1 = NeuralNet.Net(bounds).to(device)
    # weight_file_trial1 = "circles_21x21_epoch1_random_greedyo_r4_t2000_s25_rollout_diffstartloc"
    # neural_model_trial1.load_state_dict(torch.load(CONF[json_comp_conf]["neural_net_weights_path"]+weight_file_trial1))
    # neural_model_trial1.eval()

    # test_type = "trials{}_steps{}_allplanners_6".format(trials, steps)
    test_type = "trials{}_steps{}_test".format(trials, steps)
    filename = '{}planner_scores_multibot/{}'.format(
        CONF[json_comp_conf]["pickle_path"], test_type)

    debug_mcts_reward_greedy_list = list()
    debug_mcts_reward_network_list = list()

    # so we know which experiement we are running
    print("{} - {}".format(weight_file, test_type))

    mcts_plot_planners = [
        "net_fullcomm_net_fullcomm", "net_partialcomm_net_partialcomm", "net_poorcomm_net_poorcomm",
        "random_fullcomm_net_fullcomm", "random_partialcomm_net_partialcomm", "random_poorcomm_net_poorcomm",
        "greedy_fullcomm_greedy_fullcomm", "greedy_partialcomm_greedy_partialcomm", "greedy_poorcomm_greedy_poorcomm",
        "random_fullcomm_greedy_fullcomm", "random_partialcomm_greedy_partialcomm", "random_poorcomm_greedy_poorcomm"]

    test_start_time = time.time()
    for i in tqdm(range(trials)):
        trial_start_time = time.time()
        print("TRIAL NO: {}".format(i+1))
        obs_density = 18
        map = Map(bounds, obs_density, (), False)
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
                        mcts_planner = rollout_type + '_' + reward_type

                        if mcts_planner in mcts_plot_planners:

                            print("Rollout: {}, Reward: {}".format(
                                rollout_type, reward_type))

                            # this is for pickling the data
                            curr_list = score_lists[score_list]
                            if len(curr_list) == 0:
                                curr_list.append(mcts_planner)
                            score_list += 1

                            # this is for calculating the end score counting only unique seen cells
                            obs_occupied_oracle = set()
                            obs_free_oracle = set()

                            # the map has to be the same for each planner
                            for bot in robots:
                                map = Map(bounds, obs_density, copy.deepcopy(
                                    unobs_occupied), True)
                                sensor_model = SensorModel(bot, map)
                                simulator = Simulator(
                                    map, bot, sensor_model, planner, rollout_type, reward_type)
                                # start_loc = bot.get_start_loc()
                                bot.set_loc(start_loc[0], start_loc[1])
                                bot.add_map(map)
                                bot.add_sensor_model(sensor_model)
                                bot.add_simulator(simulator)
                                # this adds the initial matrices to appropriate lists
                                bot_simulator = bot.get_simulator()
                                bot_simulator.initialize_data(
                                    bots_starting_locs, obs_occupied_oracle)
                                # this is needed incase any locations are scanned in the initial position
                                obs_occupied_oracle = obs_occupied_oracle.union(
                                    bot_simulator.get_obs_occupied())
                                obs_free_oracle = obs_free_oracle.union(
                                    bot_simulator.get_obs_free())

                            steps_count = 0
                            acc_score = list()
                            for step in tqdm(range(steps)):
                                curr_robot_positions = set()

                                for bot in tqdm(robots):
                                    simulator = bot.get_simulator()
                                    sensor_model = bot.get_sensor_model()

                                    # oracle_visualize(robots, bounds, map, planner, reward_type, rollout_type)

                                    simulator.run(neural_model, curr_robot_positions, train=True,
                                                  debug_mcts_reward_greedy_list=debug_mcts_reward_greedy_list,
                                                  debug_mcts_reward_network_list=debug_mcts_reward_network_list,
                                                  device=device, CONF=CONF, json_comp_conf=json_comp_conf)

                                    # simulator.visualize(robots, step)

                                    # to keep track of score
                                    obs_occupied_oracle = obs_occupied_oracle.union(
                                        simulator.get_obs_occupied())
                                    obs_free_oracle = obs_free_oracle.union(
                                        simulator.get_obs_free())

                                step_score = len(obs_occupied_oracle)

                                # oracle_visualize(robots, bounds, map, planner, step, CONF, json_comp_conf, step_score,
                                #      show=False, save=True, rollout_type=rollout_type, reward_type=reward_type)
                                
                                if len(acc_score) > 0:
                                    acc_score.append(
                                        acc_score[-1] + step_score)
                                acc_score.append(step_score)

                                steps_count += 1
                                if rollout_type in ("random_fullcomm", "greedy_fullcomm", "net_fullcomm") or reward_type in ("greedy_fullcomm", "net_fullcomm"):
                                    communicate(robots, obs_occupied_oracle, obs_free_oracle)
                                elif rollout_type in ("random_partialcomm", "greedy_partialcomm", "net_partialcomm") and (steps_count%partial_comm_step == 0):
                                    communicate(robots, obs_occupied_oracle, obs_free_oracle)
                                elif reward_type in ("random_partialcomm", "greedy_partialcomm", "net_partialcomm") and (steps_count%partial_comm_step) == 0:
                                    communicate(robots, obs_occupied_oracle, obs_free_oracle)
                                elif rollout_type in ("random_poorcomm", "greedy_poorcomm", "net_poorcomm") and (steps_count%poor_comm_step) == 0:
                                    communicate(robots, obs_occupied_oracle, obs_free_oracle)
                                elif reward_type in ("random_poorcomm", "greedy_poorcomm", "net_poorcomm") and (steps_count%poor_comm_step) == 0:
                                    communicate(
                                        robots, obs_occupied_oracle, obs_free_oracle)
                                steps_end = time.time()

                            score = len(obs_occupied_oracle)
                            print("Score: ", score)
                            curr_list.append(score)

                            # oracle_visualize(robots, bounds, map, planner, reward_type, rollout_type)

                            # pickle progress
                            outfile = open(filename, 'wb')
                            pickle.dump(score_lists, outfile)
                            outfile.close()

            else:  # these are the myopic planners

                # adds planner name to the visualization list
                curr_list = score_lists[score_list]
                if len(curr_list) == 0:
                    curr_list.append(planner)
                score_list += 1

                # this is for calculating the end score counting only unique seen cells
                obs_occupied_oracle = set()
                obs_free_oracle = set()

                # the map has to be the same for each planner
                for bot in robots:
                    map = Map(bounds, obs_density,
                              copy.deepcopy(unobs_occupied), True)
                    sensor_model = SensorModel(bot, map)
                    simulator = Simulator(map, bot, sensor_model, planner)
                    # start_loc = bot.get_start_loc()
                    bot.set_loc(start_loc[0], start_loc[1])
                    bot.add_map(map)
                    bot.add_sensor_model(sensor_model)
                    bot.add_simulator(simulator)
                    # this adds the initial matrices to appropriate lists
                    bot_simulator = bot.get_simulator()
                    bot_simulator.initialize_data(
                        bots_starting_locs, obs_occupied_oracle)
                    # this is needed incase any locations are scanned in the initial position
                    obs_occupied_oracle = obs_occupied_oracle.union(
                        bot_simulator.get_obs_occupied())
                    # obs_free_oracle = obs_free_oracle.union(bot_simulator.get_obs_free())
                    obs_free_oracle = obs_free_oracle.union(
                        bot_simulator.get_obs_free())

                steps_count = 0
                for step in range(steps):
                    curr_robot_positions = set()

                    # run multiple robots in same map
                    for bot in robots:
                        simulator = bot.get_simulator()
                        sensor_model = bot.get_sensor_model()

                        # we run it without obs_occupied_oracle because if not the normal planners have oracle info
                        simulator.run(
                            neural_model, curr_robot_positions, train=True, device=device)

                        # simulator.visualize(robots, step)

                        # to keep track of score
                        obs_occupied_oracle = obs_occupied_oracle.union(
                            simulator.get_obs_occupied())
                        # obs_free_oracle = obs_free_oracle.union(bot_simulator.get_obs_free())
                        obs_free_oracle = obs_free_oracle.union(
                            simulator.get_obs_free())

                        # oracle_visualize(robots, bounds, map, planner)

                    step_score = len(obs_occupied_oracle)

                    steps_count += 1
                    if planner in ("random_fullcomm", "greedy_fullcomm", "net_fullcomm", "net_trial"):
                        # print("DEBUG FULL, Step {}".format(step))
                        communicate(robots, obs_occupied_oracle, obs_free_oracle)
                    elif planner in ("random_partialcomm", "greedy_partialcomm", "net_partialcomm") and (steps_count%partial_comm_step == 1):
                        # print("DEBUG PARTIAL, Step {}".format(step))
                        communicate(robots, obs_occupied_oracle, obs_free_oracle)
                    elif planner in ("random_poorcomm", "greedy_poorcomm", "net_poorcomm") and (steps_count%poor_comm_step == 0):
                        # print("DEBUG POOR, Step {}".format(step))
                        communicate(robots, obs_occupied_oracle, obs_free_oracle)
                    steps_end = time.time()

                score = len(obs_occupied_oracle)
                print("Score: ", score)
                curr_list.append(score)
                # oracle_visualize(robots, bounds, map, planner)

                # if planner == "net_nocomm" or planner == "net_partialcomm" or planner == "net_fullcomm":
                #     oracle_visualize(robots, bounds, map, planner)

                # pickle progress
                outfile = open(filename, 'wb')
                pickle.dump(score_lists, outfile)
                outfile.close()

        trial_end_time = time.time()
        print("Trial time taken (mins): ",
              (trial_end_time - trial_start_time)/60)

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
        if len(score_list) == 0:  # this condition was added because we are skipping some mcts planners
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
        if len(score_list) == 0:  # this condition was added because we are skipping some mcts planners
            continue
        score_list.remove(score_list[0])

    # do this otherwise x axis is not correct
    for i in x_pos:
        x_pos[i] += 1

    fig = plt.figure(figsize=(10, 7))
    # Creating axes instance
    plt.boxplot(score_lists_copy)
    plt.xticks(x_pos, bars, rotation=25)
    plt.title(weight_file+"_trials:"+str(trials)+"_steps:"+str(steps))
    plt.tight_layout()
    plt.show()
