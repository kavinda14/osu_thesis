import pickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    
    """
    This script serves two purposes:
    1) Check if greedy and network for mcts show an almost linear pattern
    3) To see if the ACCUMULATED reward for mcts rollout_reward converges (see google slides)
    """

    # unpickle scores
    filename = '/home/kavi/thesis/pickles/debug_reward_greedy_list'
    infile = open(filename,'rb')
    debug_reward_greedy_list = pickle.load(infile)
    infile.close()

    filename = '/home/kavi/thesis/pickles/debug_reward_network_list'
    infile = open(filename,'rb')
    debug_reward_network_list = pickle.load(infile)
    infile.close()

    # create axes for scatter plot
    y1 = debug_reward_greedy_list
    print("debug_reward_greedy_list: ", len(y1))

    y2 = debug_reward_network_list
    print("debug_reward_network_list", len(y2))

    # This is for seeing whether greedy and network are linear
    # Network should be linear with greedy because the network is trained mostly on greedy data
    # The 90-100k is there because we want to check only for a single trial
    # plt.scatter(y1, y2, s=2.0)
    # plt.xlabel("greedy")
    # plt.ylabel("network")
    # plt.show()

    # see if reward is converging
    # the for loop separates the data from a single trial into diff segments so we can boxplot
    # the increment was chose by seeing the length of the list and splitting it evenly somehow
    increment = len(debug_reward_greedy_list)//12
    start = 0
    end = increment
    data_dict = {}
    for i in range(12):
        data_dict["data_" + str(i+1)] = debug_reward_greedy_list[start:end]
        start = end
        end += increment

    fig = plt.figure(figsize =(10, 7))
    
    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])
    
    data = list()
    for data_list in data_dict.values():
        data.append(data_list)

    # create box plot 
    bp = ax.boxplot(data)
    
    # show plot
    plt.show()