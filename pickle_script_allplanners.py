import pickle
from turtle import title
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

pgf_with_latex = {
    "text.usetex": True,            # use LaTeX to write all text
    "pgf.rcfonts": False,           # Ignore Matplotlibrc
    "pgf.preamble": [
        # r'\usepackage{color}'     # xcolor for colours
        r'\usepackage[dvipsnames]{xcolor}'     # xcolor for colours
        
    ]
}
matplotlib.rcParams.update(pgf_with_latex)

'''
This pickle script is for all the greedy planners + mcts variations.
'''

if __name__ == "__main__":

    # unpickle scores
    # alienware
    # filename = '/home/kavi/thesis/pickles/planner_scores'
    # filename = '/home/kavi/thesis/pickles/planner_scores_multibot/trial10_steps25_roll_random_greedy_rew_random_greedy_net_everystep'
    # filename = '/home/kavi/thesis/pickles/planner_scores_multibot/trial100_steps40_roll_random_greedy_net_everystep_rew_greedy_net_everystep'
    filename = '/home/kavi/thesis/pickles/planner_scores_multibot/trial100_steps25_comm_nocomm'
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
        if len(score_list) == 0: # this condition was added because we are skipping some mcts planners
            score_lists.remove(score_list)
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
        if len(score_list) == 0: # this condition was added because we are skipping some mcts planners
            score_lists_copy.remove(score_list)
            continue
        score_list.remove(score_list[0])   

    # swapping of plots
    temp = score_lists_copy[15]
    score_lists_copy[15] = score_lists_copy[14]
    score_lists_copy[14] = temp

    temp = score_lists_copy[8]
    score_lists_copy[8] = score_lists_copy[10]
    score_lists_copy[10] = temp

    temp = score_lists_copy[9]
    score_lists_copy[9] = score_lists_copy[10]
    score_lists_copy[10] = temp             
    

    # do this otherwise x axis is not correct
    for i in x_pos:
        x_pos[i] += 1

    # x labels 
    planner_names = [r'\textcolor{black}{random}', r'\textcolor{black}{greedy-o}', r'\textcolor{black}{greedy-o}', r'\textcolor{black}{greedy-no}',
    r'\textcolor{black}{greedy-no}', r'\textcolor{black}{network}', r'\textcolor{black}{network}',
    r'\textcolor{red}{random}' + r' ' + r'\textcolor{blue}{greedy}', r'\textcolor{red}{greedy}' + r' ' + r'\textcolor{blue}{greedy}',
    r'\textcolor{red}{random}' + r' ' + r'\textcolor{blue}{network}', r'\textcolor{red}{random}' + r' ' + r'\textcolor{blue}{network}',
    r'\textcolor{red}{greedy}' + r' ' + r'\textcolor{blue}{network}', r'\textcolor{red}{greedy}' + r' ' + r'\textcolor{blue}{network}',
    r'\textcolor{red}{network}' + r' ' + r'\textcolor{blue}{greedy}', r'\textcolor{red}{network}' + r' ' + r'\textcolor{blue}{greedy}',
    r'\textcolor{red}{network}' + r' ' + r'\textcolor{blue}{network}',
    r'\textcolor{red}{network}' + r' ' + r'\textcolor{blue}{network}']

    # legends
    green_patch = mpatches.Patch(color='red', label='mcts rollout')
    blue_patch = mpatches.Patch(color='blue', label='mcts reward')
    pink_patch = mpatches.Patch(color='pink', label='communication - every step')
    orange_patch = mpatches.Patch(color='orange', label='communication - every third step')


    # caption = "Figure: For all planners shown above, the robots are communicating after every robot has made a step"
    fig = plt.figure(figsize =(10, 7))
    
    box = plt.boxplot(score_lists_copy, patch_artist=True)
    colors = ['pink', 'orange', 'pink', 'orange', 'pink', 'orange',
              'pink', 'pink', 'pink', 'orange', 'pink', 'orange',
              'pink', 'orange', 'pink', 'orange', 'pink']
    
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # plt.xticks(x_pos, bars, rotation=26)
    plt.xticks(x_pos, planner_names, rotation=26)
    plt.axvline(x=7.5)
    plt.title("Greedy Planners vs MCTS Planners - 4 robots - 25 steps/robot - 17 trials(maps)")
    temp = 'greedy'
    # plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
    # plt.tight_layout() # does not work when using LaTex, add it when doing .show()    
    plt.text(2.0, 110.0, "greedy planners")
    plt.text(8.5, 110.0, "mcts planners")
    plt.legend(handles=[pink_patch, orange_patch,green_patch, blue_patch], loc="lower right")
    
    plt.ylabel("Total Reward")
    # plt.show()
    plt.savefig("test_allplanners3.pdf")