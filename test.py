from SensorModel import SensorModel
from Map import Map
from Robot import Robot
from Simulator import Simulator
import copy
import matplotlib.pyplot as plt
import numpy as np
import random as r
import time as time
import pickle
import NeuralNet
import torch
from tqdm import tqdm
from basic_MCTS_python.reward import reward_greedy

import cProfile
import pstats

if __name__ == "__main__":

    # Bounds need to be an odd number for the action to always be in the middle
    # greedy-o: greedy oracle (knows where the obstacles are in map)
    # greedy-no: greedy non-oracle (counts total unobserved cells in map)
    # planner_options = ["random", "greedy-o", "greedy-no", "network", "mcts"]
    planner_options = ["random", "greedy-o", "greedy-no", "network"]
    # planner_options = ["mcts"]
    rollout_options = ["random", "greedy", "network"]
    # rollout_options = ["network"]
    reward_options = ["random", "greedy", "network"]
    # reward_options = ["network"]
    bounds = [21, 21]
    trials = 100
    steps = 60
    visualize = True
    # profiling functions
    profile = False

    if profile:
        pr = cProfile.Profile()
        pr.enable()

    # this is for pickling and visualizing the data -> check pickle_script.py
    if "mcts" in planner_options:
        score_lists = [list() for _ in range((len(planner_options)-1)+(len(rollout_options)*len(reward_options)))]
    else:
        score_lists = [list() for _ in range(len(planner_options))]
    
    # load neural net
    # weight_file = "circles_21x21_epoch3_random_greedyno_t800_s200_rollout"
    weight_file = "circles_21x21_epoch10_random_greedy-no_t300_s9000"
    neural_model = NeuralNet.Net(bounds)
    # neural_model.load_state_dict(torch.load("/home/kavi/thesis/neural_net_weights/circles_random_21x21_epoch2_random_greedyo_greedyno_t500_s200"))
    # neural_model.load_state_dict(torch.load("/home/kavi/thesis/neural_net_weights/circles_21x21_epoch3_random_t600_s1000"))    
    neural_model.load_state_dict(torch.load("/home/kavi/thesis/neural_net_weights/"+weight_file))    
    neural_model.eval()

    # this is for pickling the score_lists
    filename = '/home/kavi/thesis/pickles/planner_scores_test'

    test_start_time = time.time()
    for i in tqdm(range(trials)):
        trial_start_time = time.time()
        print("TRIAL NO: {}".format(i+1))
        map = Map(bounds, 7, (), False)
        unobs_occupied = copy.deepcopy(map.get_unobs_occupied())
        
        # for pickling data
        score_list = 0

        # start robot at random valid location on map
        valid_starting_loc = False
        while not valid_starting_loc:
            x = r.randint(0, bounds[0]-1)
            y = r.randint(0, bounds[0]-1)
            valid_starting_loc = map.check_loc(x, y)

        for planner in planner_options:
            print("Planner: {}".format(planner))

            if planner == "mcts":
                for rollout_type in rollout_options:
                    for reward_type in reward_options:
                        print("Rollout: {}, Reward: {}".format(rollout_type, reward_type))
                        # this is for pickling the data
                        curr_list = score_lists[score_list]
                        if len(curr_list) == 0:
                            curr_list.append(rollout_type + '_' + reward_type)
                        score_list += 1

                        # map object is here as well just to reset the map to its initial state
                        map = Map(bounds, 7, copy.deepcopy(unobs_occupied), True)
                        robot = Robot(x, y, bounds, map)
                        # robot = Robot(9, 1, bounds, map)
                        sensor_model = SensorModel(robot, map)
                        start = time.time()
                        simulator = Simulator(map, robot, sensor_model, planner, rollout_type, reward_type)
                        if visualize:
                            simulator.visualize()
                        simulator.run(steps, neural_model)
                        end = time.time()
                        if visualize:
                            simulator.visualize()
                        score = sum(sensor_model.get_final_scores())                        
                        curr_list.append(score)
                        
                        print("Score: ", score)
                        print("Time taken (secs): ", end - start)
                        print()
                        
                        # pickle progress
                        outfile = open(filename,'wb')
                        pickle.dump(score_lists, outfile)
                        outfile.close()


            else: # these are the myopic planners
                curr_list = score_lists[score_list]
                if len(curr_list) == 0:
                    curr_list.append(planner)
                score_list += 1
                
                map = Map(bounds, 7, copy.deepcopy(unobs_occupied), True)
                robot = Robot(x, y, bounds, map)
                # robot = Robot(9, 1, bounds, map)
                sensor_model = SensorModel(robot, map)
                start = time.time()
                simulator = Simulator(map, robot, sensor_model, planner)
                if visualize:
                    simulator.visualize()
                simulator.run(steps, neural_model)
                end = time.time()
                if visualize:
                    simulator.visualize()
                score = sum(sensor_model.get_final_scores())     
                print("Score: ", score)
                print("Time taken (secs): ", end - start)
                print()
                
                curr_list.append(score)
                
                # pickle progress
                outfile = open(filename,'wb')
                pickle.dump(score_lists, outfile)
                outfile.close()

        trial_end_time = time.time()
        print("Trial time taken (mins): ", (trial_end_time - trial_start_time)/60)
    
    test_end_time = time.time()
    print("Total time taken (mins): ", (test_end_time - test_start_time)/60)

    if profile:
        pr.disable()
        pr.print_stats()
        with open("cProfile_stats.txt", "w") as f:
            ps = pstats.Stats(pr, stream=f)
            ps.sort_stats('cumtime')
            ps.print_stats()

    ## Create Bar Graphs
    bars = list()
    scores = list()

    for score_list in score_lists:
        planner_name = score_list[0]
        bars.append(planner_name)
        del score_list[0]
        curr_score = sum(score_list)/len(score_list)
        scores.append(curr_score)

    x_pos = np.arange(len(bars))
    # plt.bar(x_pos, scores, color=['#33e6ff', 'red', 'green', 'blue', '#FFC0CB', '#800080', '#fdbe83', '#00ab66', '#0b1320', '#ddceff', '#4000ff', '#ff876f', '#540077'])
    # plt.xticks(x_pos, bars, rotation=45)
    # plt.title(weight_file)
    
    # # puts the value on top of each bar
    # for i in range(len(bars)):
    #     plt.text(i, scores[i], scores[i], ha = 'center')

    # plt.show()

    # Box plot
    score_lists_copy = score_lists
    for score_list in score_lists_copy:
        score_list.remove(score_list[0])   

    # do this otherwise x axis is not correct
    for i in x_pos:
        x_pos[i] += 1

    fig = plt.figure(figsize =(10, 7))
    plt.boxplot(score_lists_copy)
    plt.xticks(x_pos, bars, rotation=45)
    plt.title(weight_file+"_trials:"+str(trials)+"_steps:"+str(steps))
    plt.show()



