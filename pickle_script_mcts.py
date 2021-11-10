import pickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    

    # unpickle scores
    filename = '/home/kavi/thesis/pickles/debug_reward_greedy_list'
    infile = open(filename,'rb')
    debug_reward_greedy_list = pickle.load(infile)
    infile.close()

    filename = '/home/kavi/thesis/pickles/debug_reward_network_list'
    infile = open(filename,'rb')
    debug_reward_network_list = pickle.load(infile)
    infile.close()

    abs_max1 = np.amax(np.abs(debug_reward_greedy_list))
    print('abs_max1', abs_max1)
    normalized_array_greedy = np.asarray(debug_reward_greedy_list) * (5.0 / abs_max1)

    abs_max2 = np.amax(np.abs(debug_reward_network_list))
    print(abs_max2)
    normalized_array_network = np.asarray(debug_reward_network_list) * (5.0 / abs_max2)

    print('debug_reward_greedy_list', len(normalized_array_greedy))
    print('debug_reward_network_list', normalized_array_network)

    # y1 = debug_reward_greedy_list
    # x1 = np.array([i for i in range(len(y1))])

    # y2 = debug_reward_network_list
    # x2 = np.array([i for i in range(len(y2))])

    # plt.scatter(x1[90000:100000], y1[90000:100000], s=2.0, label='greedy')
    # plt.scatter(x2[90000:100000], y2[90000:100000], s=2.0, label="network")
    # plt.legend(loc='best')
    # plt.show()

    # plt.scatter(y1[90000:100000], y2[90000:100000], s=2.0)
    # plt.xlabel("greedy")
    # plt.ylabel("network")
    # plt.show()

    # plt.scatter(y1[0:10000], y2[0:10000], s=2.0)
    # plt.scatter(y1[10000:20000], y2[10000:20000], s=2.0)
    # plt.scatter(y1[20000:30000], y2[20000:30000], s=2.0)
    # plt.xlabel("greedy")
    # plt.ylabel("network")
    # plt.show()

    
    increment = 10000
    # data_1 = normalized_array_network[0:10000]
    # data_2 = normalized_array_network[0+increment:10000+increment]
    # data_3 = normalized_array_network[0+(increment*2):10000+(increment*2)]
    # data_4 = normalized_array_network[0+(increment*3):10000+(increment*3)]
    # data_5 = normalized_array_network[0+(increment*4):10000+(increment*4)]
    # data_6 = normalized_array_network[0+(increment*5):10000+(increment*5)]
    # data_7 = normalized_array_network[0+(increment*6):10000+(increment*6)]
    # data_8 = normalized_array_network[0+(increment*7):10000+(increment*7)]
    # data_9 = normalized_array_network[0+(increment*8):10000+(increment*8)]
    # data_10 = normalized_array_network[0+(increment*9):10000+(increment*9)]
    # data = [data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10]
    # data = [data_10]

    increment2 = 1000
    data_1 = normalized_array_network[(increment*9):(increment*9)+(increment2)]
    data_2 = normalized_array_network[(increment*9):(increment*9)+(increment2*2)]
    data_3 = normalized_array_network[(increment*9):(increment*9)+(increment2*3)]
    data_4 = normalized_array_network[(increment*9):(increment*9)+(increment2*4)]
    data_5 = normalized_array_network[(increment*9):(increment*9)+(increment2*5)]
    data_6 = normalized_array_network[(increment*9):(increment*9)+(increment2*6)]
    data_7 = normalized_array_network[(increment*9):(increment*9)+(increment2*7)]
    data_8 = normalized_array_network[(increment*9):(increment*9)+(increment2*8)]
    data_9 = normalized_array_network[(increment*9):(increment*9)+(increment2*9)]
    data_10 = normalized_array_network[(increment*9):(increment*9)+(increment2*10)]
    data = [data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10]

    fig = plt.figure(figsize =(10, 7))
    
    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Creating plot
    bp = ax.boxplot(data)
    
    # show plot
    plt.show()