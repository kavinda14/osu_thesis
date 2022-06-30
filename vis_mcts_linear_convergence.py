from utils import get_CONF, get_json_comp_conf
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale

if __name__ == "__main__":
    
    """
    This script serves two purposes:
    1) Check if greedy and network for mcts show an almost linear pattern
    3) To see if the ACCUMULATED reward for mcts rollout_reward converges (see google slides)
    """

    CONF = get_CONF()
    json_comp_conf = get_json_comp_conf()

    # unpickle scores
    filename = CONF[json_comp_conf]["pickle_path"] + "debug_reward_greedy_list"
    infile = open(filename,'rb')
    debug_reward_greedy_list = pickle.load(infile)
    infile.close()

    filename = CONF[json_comp_conf]["pickle_path"] + "debug_reward_network_list"
    infile = open(filename, 'rb')
    debug_reward_network_list = pickle.load(infile)
    infile.close()

    # scatter plot
    y1 = debug_reward_greedy_list
    y1 = minmax_scale(y1)
    print("debug_reward_greedy_list: ", len(y1))

    y2 = debug_reward_network_list
    y2 = minmax_scale(y2)
    print("debug_reward_network_list", len(y2))

    """  
    This is for seeing whether greedy and network are linear
    Network should be linear with greedy because the network is trained mostly on greedy data
    The 90-100k is there because we want to check only for a single trial 
    """
    plt.scatter(y1, y2, s=2.0)
    plt.xlabel("greedy")
    plt.ylabel("network")
    # plt.show()

    """ 
    Reward convergence:
    the for loop separates the data from a single trial into diff segments so we can boxplot
    the increment was chose by seeing the length of the list and splitting it evenly somehow
    Remember that we need to check this for a single mcts run and not over all steps as the reward..
    ..converges in a single mcts run and not have all steps.
     """

    increment = 50
    start = 0
    end = increment
    data_dict = {}

    # modified_list = debug_reward_greedy_list[0:1000]
    # print(debug_reward_greedy_list)
    # modified_list = debug_reward_network_list[0:12000]
    modified_list = debug_reward_network_list[0:1000]

    y1 = [i for i in range(len(modified_list))]
    y2 = modified_list

    plt.scatter(y1, y2, s=2.0)
    plt.xlabel("iteration")
    plt.ylabel("score")
    plt.show()

    # print(np.average(debug_reward_greedy_list))

    # for i in range(20):
    #     data_dict["data_" + str(i+1)] = modified_list[start:end]
    #     start = end
    #     end += increment

    # # print("data_dict: ", data_dict.keys())

    # data = list()
    # for data_list in data_dict.values():
    #     data.append(data_list)
    # print(len(data[0]))

    # create box plot 
    # fig = plt.figure(figsize =(10, 7))
    # plt.boxplot(data)
    # plt.show()