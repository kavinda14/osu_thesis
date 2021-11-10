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

from basic_MCTS_python.reward import reward_greedy

if __name__ == "__main__":
 
    # Bounds need to be an odd number for the action to always be in the middle
    # greedy-o: greedy oracle (knows where the obstacles are in map)
    # greedy-no: greedy non-oracle (counts total unobserved cells in map)
    # planner_options = ["random", "greedy-o", "greedy-no", "network", "mcts"]
    # planner_options = ["greedy-no"]
    planner_options = ["mcts"]
    # rollout_options = ["random", "greedy", "network"]
    rollout_options = ["greedy"]
    # reward_options = ["random", "greedy", "network"]
    reward_options = ["network"]
    bounds = [21, 21]
    trials = 1
    steps = 10
    visualize = True

    # 13 because we have 13 diff planners
    # score_lists = [list() for _ in range(1)]
    score_lists = [list() for _ in range(1)]
    score_list = 0

    # load neural net
    neural_model = NeuralNet.Net(bounds)
    neural_model.load_state_dict(torch.load("/home/kavi/thesis/neural_net_weights/circles_random_21x21_epoch2_mctsrolloutdata2"))
    neural_model.eval()

    # this is for pickling the score_lists
    filename = '/home/kavi/thesis/pickles/planner_scores'

    test_start_time = time.time()
    for planner in planner_options:
        # CHECK IF MAP IN CORRECT PLACE IN Test.py
        map = Map(bounds, 7, (), False)
        unobs_occupied = copy.deepcopy(map.get_unobs_occupied())

        valid_starting_loc = False
        while not valid_starting_loc:
            x = r.randint(0, bounds[0]-1)
            y = r.randint(0, bounds[0]-1)
            valid_starting_loc = map.check_loc(x, y) 

        if planner == "mcts":
            for rollout_type in rollout_options:
                for reward_type in reward_options:
                    print("Planner: {}".format(planner))
                    print("Rollout: {}, Reward: {}".format(rollout_type, reward_type))
                    curr_list = score_lists[score_list]
                    curr_list.append(rollout_type + '_' + reward_type)
                    score_list += 1

                    for i in range(trials):
                        print("Trial no: {}".format(i))
                        map = Map(bounds, 7, copy.deepcopy(unobs_occupied), True)
                        # robot = Robot(x, y, bounds, map)
                        robot = Robot(10, 7, bounds, map)
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
                        
                        print(sensor_model.get_final_path())     
                        print("Score: ", score)
                        print("Time taken (secs): ", end - start)
                        print()
                        
                        # pickle progress
                        outfile = open(filename,'wb')
                        pickle.dump(score_lists, outfile)
                        outfile.close()


        else:
            print("Planner: {}".format(planner))
            curr_list = score_lists[score_list]
            curr_list.append(planner)
            score_list += 1

            for i in range(trials):
                print("Trial no: {}".format(i))
                map = Map(bounds, 7, copy.deepcopy(unobs_occupied), True)
                robot = Robot(x, y, bounds, map)
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
                print(sensor_model.get_final_path())                   
                print("Score: ", score)
                print("Time taken (secs): ", end - start)
                print()
                
                curr_list.append(score)
                
                # pickle progress
                outfile = open(filename,'wb')
                pickle.dump(score_lists, outfile)
                outfile.close()
    
    test_end_time = time.time()
    print("Total time taken (mins): ", (test_end_time - test_start_time)/60)

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
    plt.bar(x_pos, scores, color=['#33e6ff', 'red', 'green', 'blue', '#FFC0CB', '#800080', '#fdbe83', '#00ab66', '#0b1320'])
    plt.xticks(x_pos, bars, rotation=45)
    plt.show()

    ## Create Line Graphs
    # plt.plot(x1, random, label = "random")
    # plt.plot(x1, greedy, label = "greedy")
    # plt.plot(x1, network, label = "network")
    # plt.plot(x1, mcts, label = "mcts")

    # plt.xlabel('Trial no')
    # # Set the y axis label of the current axis.
    # plt.ylabel('Score')
    # # Set a title of the current axes.
    # plt.title('Avg scores: random: {}, greedy: {}, network: {}, mcts {}'.format(avg_random, avg_greedy, avg_network, avg_mcts))
    # # plt.title('Avg scores: random: {}, greedy: {}, network: {}'.format(avg_random, avg_greedy, avg_network))
    # # show a legend on the plot
    # plt.legend()
    # # Display a figure.
    # plt.show()


