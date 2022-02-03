import pickle
from turtle import title
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

if __name__ == "__main__":

    # unpickle scores
    # alienware
    # filename = '/home/kavi/thesis/pickles/planner_scores'
    # filename = '/home/kavi/thesis/pickles/planner_scores_multibot/trial10_steps25_roll_random_greedy_rew_random_greedy_net_everystep'
    filename = '/home/kavi/thesis/pickles/planner_scores_multibot/trial100_steps25_comm_nocomm_commstep3'
    # filename = '/home/kavi/thesis/pickles/planner_scores_multibot/test'
    # macbook
    # filename = '/Users/kavisen/osu_thesis/pickles/planner_scores_test'
    infile = open(filename,'rb')
    score_lists = pickle.load(infile)
    infile.close()

    # print(score_lists)

    ## Bar graphs
    bars = list()
    scores = list()

    for score_list in score_lists:
        if len(score_list) == 0:
            del score_list
            continue
        planner_name = score_list[0]
        print(planner_name)
        bars.append(planner_name)
        del score_list[0]
        curr_score = sum(score_list)/len(score_list)
        scores.append(curr_score)

    x_pos = np.arange(len(bars))
    # plt.bar(x_pos, scores, color=['#33e6ff', 'red', 'green', 'blue', '#FFC0CB', '#800080', '#fdbe83', '#00ab66', '#0b1320', '#ddceff'])
    # plt.xticks(x_pos, bars, rotation=45)

    # # puts the value on top of each bar
    # for i in range(len(bars)):
    #     plt.text(i, scores[i], scores[i], ha = 'center')

    # plt.show()

    # Box plot
    score_lists_copy = score_lists
    for score_list in score_lists_copy:
        if len(score_list) == 0:
            del score_list
            continue
        score_list.remove(score_list[0])   

    # do this otherwise x axis is not correct
    for i in x_pos:
        x_pos[i] += 1

    caption = "Figure: For all planners shown above, the robots are communicating after every robot has made a step"
    fig = plt.figure(figsize =(10, 7))
    plt.boxplot(score_lists_copy)
    plt.xticks(x_pos, bars, rotation=26)
    # plt.xticks(x_pos, planner_names, rotation=26)
    # plt.axvline(x=4.5)
    plt.title("Greedy Planners vs MCTS Planners")
    temp = 'greedy'
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
    plt.tight_layout() # does not work when using LaTex, add it when doing .show()    
    # plt.text(2.0, 125.0, "greedy planners")
    # plt.text(6.5, 125.0, "mcts planners")
    # plt.legend(handles=[green_patch, blue_patch], loc="lower right")
    
    plt.ylabel("Total Reward")
    plt.show()
    # plt.savefig("test.pdf")