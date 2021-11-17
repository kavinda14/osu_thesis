import pickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # unpickle scores
    filename = '/home/kavi/thesis/pickles/planner_scores'
    infile = open(filename,'rb')
    score_lists = pickle.load(infile)
    infile.close()

    print(score_lists)

    ## Bar graphs
    bars = list()
    scores = list()

    for score_list in score_lists:
        planner_name = score_list[0]
        bars.append(planner_name)
        del score_list[0]
        curr_score = sum(score_list)/len(score_list)
        scores.append(curr_score)

    x_pos = np.arange(len(bars))
    plt.bar(x_pos, scores, color=['#33e6ff', 'red', 'green', 'blue', '#FFC0CB', '#800080', '#fdbe83', '#00ab66', '#0b1320', '#ddceff'])

    plt.xticks(x_pos, bars, rotation=45)
    plt.show()

    ## Box plot
    # score_lists_copy = score_lists
    # for score_list in score_lists_copy:
    #     score_list.remove(score_list[0])   

    # # do this otherwise x axis is not correct
    # for i in x_pos:
    #     x_pos[i] += 1

    # fig = plt.figure(figsize =(10, 7))
    # plt.boxplot(score_lists_copy)
    # plt.xticks(x_pos, bars, rotation=45)
    # plt.show()