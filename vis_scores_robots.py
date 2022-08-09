# Adapted from https://www.geeksforgeeks.org/how-to-create-boxplots-by-group-in-matplotlib/
from cmath import pi
import matplotlib.pyplot as plt
import numpy as np
import pickle
from utils import get_CONF, get_json_comp_conf
import os

'''

This script is for visualizing boxplots for increasing number of robots with 
MCTS (Random) CNN and MCTS (Random) Sensor Coverage.

'''

# get the file paths from the .json config
CONF = get_CONF()
json_comp_conf = get_json_comp_conf()

# get the length of shared_files folder
_, _, files = next(os.walk(CONF[json_comp_conf]["shared_files_path"]))
# file_count = len(files)
file_count = 10

# this holds all the pickle data
pickles_dict_1 = dict()
pickles_dict_2 = dict()
pickles_dict_3 = dict()
# full_comms_planners_1 = ["2_MCTS_random_network_full", "4_MCTS_random_network_full", "6_MCTS_random_network_full", 
                        #  "8_MCTS_random_network_full", "10_MCTS_random_network_full", "12_MCTS_random_network_full"]
# partial_comms_planners_1 = ["2_MCTS_random_network_partial", "4_MCTS_random_network_partial", "6_MCTS_random_network_partial",
                            # "8_MCTS_random_network_partial", "10_MCTS_random_network_partial", "12_MCTS_random_network_partial"]
poor_comms_planners_1 = ["2_MCTS_random_network_poor", "4_MCTS_random_network_poor", "6_MCTS_random_network_poor",
                         "8_MCTS_random_network_poor", "10_MCTS_random_network_poor", "12_MCTS_random_network_poor",
                         "14_MCTS_random_network_poor", "16_MCTS_random_network_poor", "18_MCTS_random_network_poor",
                         "20_MCTS_random_network_poor", "22_MCTS_random_network_poor", "24_MCTS_random_network_poor",
                         "26_MCTS_random_network_poor"]

# full_comms_planners_2 = ["2_CellCountPlanner_fullnet", "4_CellCountPlanner_fullnet", "6_CellCountPlanner_fullnet", "8_CellCountPlanner_fullnet", "10_CellCountPlanner_fullnet"]
# partial_comms_planners_2 = ["2_CellCountPlanner_partialnet", "4_CellCountPlanner_partialnet", "6_CellCountPlanner_partialnet",
                            # "8_CellCountPlanner_partialnet", "10_CellCountPlanner_partialnet", "12_CellCountPlanner_partialnet"]
poor_comms_planners_2 = ["2_CellCountPlanner_poornet", "4_CellCountPlanner_poornet", "6_CellCountPlanner_poornet",
                         "8_CellCountPlanner_poornet", "10_CellCountPlanner_poornet", "12_CellCountPlanner_poornet",
                         "14_CellCountPlanner_poornet", "16_CellCountPlanner_poornet", "18_CellCountPlanner_poornet",
                         "20_CellCountPlanner_poornet", "22_CellCountPlanner_poornet", "24_CellCountPlanner_poornet",
                         "26_CellCountPlanner_poornet"]

# full_comms_planners_3 = ["2_MCTS_random_cellcount_full", "4_MCTS_random_cellcount_full", "6_MCTS_random_cellcount_full",
                        #  "8_MCTS_random_cellcount_full", "10_MCTS_random_cellcount_full", "12_MCTS_random_cellcount_full"]
# partial_comms_planners_3 = ["2_MCTS_random_cellcount_partial", "4_MCTS_random_cellcount_partial", "6_MCTS_random_cellcount_partial",
                            # "8_MCTS_random_cellcount_partial", "10_MCTS_random_cellcount_partial", "12_MCTS_random_cellcount_partial"]
poor_comms_planners_3 = ["2_MCTS_random_cellcount_poor", "4_MCTS_random_cellcount_poor", "6_MCTS_random_cellcount_poor",
                         "8_MCTS_random_cellcount_poor", "10_MCTS_random_cellcount_poor", "12_MCTS_random_cellcount_poor", 
                         "14_MCTS_random_cellcount_poor", "16_MCTS_random_cellcount_poor", "18_MCTS_random_cellcount_poor",
                         "20_MCTS_random_cellcount_poor", "22_MCTS_random_cellcount_poor", "24_MCTS_random_cellcount_poor",
                         "26_MCTS_random_cellcount_poor"]

# all_planners_1 = full_comms_planners_1 + partial_comms_planners_1 + poor_comms_planners_1
# all_planners_1 = partial_comms_planners_1
all_planners_1 = poor_comms_planners_1
# all_planners_2 = full_comms_planners_2 + partial_comms_planners_2 + poor_comms_planners_2
# all_planners_2 = partial_comms_planners_2
all_planners_2 = poor_comms_planners_2
# all_planners_3 = full_comms_planners_3 + partial_comms_planners_3 + poor_comms_planners_3
# all_planners_3 = partial_comms_planners_3
all_planners_3 = poor_comms_planners_3

for planner in all_planners_1:
    pickles_dict_1[planner] = list()   

for planner in all_planners_2:
    pickles_dict_2[planner] = list()

for planner in all_planners_3:
    pickles_dict_3[planner] = list()

for i in range(file_count):
    # filename = CONF[json_comp_conf]["shared_files_path"] +  "scores_r4_t100_s50_{}".format(i+1)
    # filename = CONF[json_comp_conf]["shared_files_path"] +  "scores_circularworld_r4_t100_s20_{}".format(i+1)
    # filename = CONF[json_comp_conf]["shared_files_path"] + "scores_depoeworld_r4_t30_s20_test3"
    filename = CONF[json_comp_conf]["shared_files_path"] + "scores_depoeworld_r4_t20_s100_{}".format(i+1)

    infile = open(filename, 'rb')
    temp_dict = pickle.load(infile)

    # append values for each planner
    for key in temp_dict.keys():
        if key in pickles_dict_1:
            pickles_dict_1[key] += temp_dict[key]
    
    for key in temp_dict.keys():
        if key in pickles_dict_2:
            pickles_dict_2[key] += temp_dict[key]

    for key in temp_dict.keys():
        if key in pickles_dict_3:
            pickles_dict_3[key] += temp_dict[key]

    infile.close()


# combine the pickles_dict planners in the dicts given above
# full_comms_dict_1 = {k: list() for k in full_comms_planners_1}
# partial_comms_dict_1 = {k: list() for k in partial_comms_planners_1}
poor_comms_dict_1 = {k: list() for k in poor_comms_planners_1}

# full_comms_dict_2 = {k: list() for k in full_comms_planners_2}
# partial_comms_dict_2 = {k: list() for k in partial_comms_planners_2}
poor_comms_dict_2 = {k: list() for k in poor_comms_planners_2}

# full_comms_dict_3 = {k: list() for k in full_comms_planners_3}
# partial_comms_dict_3 = {k: list() for k in partial_comms_planners_3}
poor_comms_dict_3 = {k: list() for k in poor_comms_planners_3}

# for key in full_comms_dict_1.keys():
#     if key in pickles_dict_1.keys():
#         full_comms_dict_1[key] += pickles_dict_1[key]
# for key in partial_comms_dict_1.keys():
#     if key in pickles_dict_1.keys():
#         partial_comms_dict_1[key] += pickles_dict_1[key]
for key in poor_comms_dict_1.keys():
    if key in pickles_dict_1.keys():
        poor_comms_dict_1[key] += pickles_dict_1[key]

# for key in full_comms_dict_2.keys():
#     if key in pickles_dict_2.keys():
#         full_comms_dict_2[key] += pickles_dict_2[key]
# for key in partial_comms_dict_2.keys():
#     if key in pickles_dict_2.keys():
#         partial_comms_dict_2[key] += pickles_dict_2[key]
for key in poor_comms_dict_2.keys():
    if key in pickles_dict_2.keys():
        poor_comms_dict_2[key] += pickles_dict_2[key]

# for key in full_comms_dict_3.keys():
#     if key in pickles_dict_3.keys():
#         full_comms_dict_3[key] += pickles_dict_3[key]
# for key in partial_comms_dict_3.keys():
#     if key in pickles_dict_3.keys():
#         partial_comms_dict_3[key] += pickles_dict_3[key]
for key in poor_comms_dict_3.keys():
    if key in pickles_dict_3.keys():
        poor_comms_dict_3[key] += pickles_dict_3[key]


# create the score lists for the boxplot                
# results_full_comms_1 = [np.array(value) for value in full_comms_dict_1.values()]
# results_partial_comms_1 = [np.array(value) for value in partial_comms_dict_1.values()]
results_poor_comms_1 = [np.array(value) for value in poor_comms_dict_1.values()]
# results_full_comms_2 = [np.array(value) for value in full_comms_dict_2.values()]
# results_partial_comms_2 = [np.array(value) for value in partial_comms_dict_2.values()]
results_poor_comms_2 = [np.array(value) for value in poor_comms_dict_2.values()]
# results_full_comms_3 = [np.array(value) for value in full_comms_dict_2.values()]
# results_partial_comms_3 = [np.array(value) for value in partial_comms_dict_3.values()]
results_poor_comms_3 = [np.array(value) for value in poor_comms_dict_3.values()]

# ticks = ['2', '4', '6', '8', '10', '12', '14', '16', '18', '20', '22', '24', '26']
ticks = ['2', '4', '6', '8', '10', '12', '14', '16']
# Do the plots
# full_comms_plot_1 = plt.boxplot(results_full_comms_1,
                # positions=np.array(np.arange(len(results_full_comms_1)))*4.0-1.2,widths=0.5, patch_artist=True,)

# partial_comms_plot_1 = plt.boxplot(results_partial_comms_1,
                # positions=np.array(np.arange(len(results_partial_comms_1)))*4.0-0.6,widths=0.5, patch_artist=True,)

poor_comms_plot_1 = plt.boxplot(results_poor_comms_1,
                                positions=np.array(np.arange(len(results_poor_comms_1)))*4.0-0.8, widths=0.7, patch_artist=True)



# full_comms_plot_2 = plt.boxplot(results_full_comms_2,
                # positions=np.array(np.arange(len(results_full_comms_2)))*4.0+0.6, widths=0.5, patch_artist=True,)

# partial_comms_plot_2 = plt.boxplot(results_partial_comms_2,
                # positions=np.array(np.arange(len(results_partial_comms_2)))*4.0+0.0,widths=0.5, patch_artist=True,)

poor_comms_plot_2 = plt.boxplot(results_poor_comms_2,
                positions=np.array(np.arange(len(results_poor_comms_2)))*4.0+0.0,widths=0.7, patch_artist=True)

# full_comms_plot_3 = plt.boxplot(results_full_comms_3,
                # positions=np.array(np.arange(len(results_full_comms_3)))*4.0+0.6, widths=0.5, patch_artist=True,)

# partial_comms_plot_3 = plt.boxplot(results_partial_comms_3,
                # positions=np.array(np.arange(len(results_partial_comms_3)))*4.0+0.6,widths=0.5, patch_artist=True,)

poor_comms_plot_3 = plt.boxplot(results_poor_comms_3,
                positions=np.array(np.arange(len(results_poor_comms_3)))*4.0+0.8,widths=0.7, patch_artist=True)

# full comm medians
# print("full_comm_medians: ")
# [print(item.get_ydata()[1]) for item in full_comms_plot['medians']]
# print()

# # partial comm medians
# print("partial_comm_medians: ")
# [print(item.get_ydata()[1]) for item in partial_comms_plot['medians']]
# print()

# # poor comm medians
# print("poor_comm_medians: ")
# [print(item.get_ydata()[1]) for item in poor_comms_plot['medians']]
# print()

# Do formatting
font_size = 30

def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color='k')
    for b in plot_name['boxes']:
    	b.set_facecolor(color_code)
         
    # use plot function to draw a small line to name the legend.
    plt.plot([], 's', c='w',mec='k', mfc=color_code, label=label, ms="20")
    plt.legend(loc='lower right', fontsize=font_size)
 
 
# Setting colors for each groups
# define_box_properties(full_comms_plot_1, '#55FF22', 'Full')
# define_box_properties(full_comms_plot_2, '#55FF22', 'Full')
# define_box_properties(full_comms_plot_3, '#55FF22', 'Full')
# define_box_properties(partial_comms_plot_1, '#ffd522', 'MCTS (Random) CNN')
# define_box_properties(partial_comms_plot_2, '#ff6622', 'Greedy CNN')
# define_box_properties(partial_comms_plot_3, '#ff224d', 'MCTS (Random) Sensor Coverage')
define_box_properties(poor_comms_plot_1, '#ffd522', 'MCTS (Random) CNN')
define_box_properties(poor_comms_plot_2, '#ff6622', 'Greedy CNN')
define_box_properties(poor_comms_plot_3, '#ff224d', 'MCTS (Random) Sensor Coverage')
 
# Other formatting
plt.xticks(np.arange(0, len(ticks) * 4, 4), ticks, fontsize=font_size)
plt.xlim(-1.5, len(ticks)*4-2.5)
 
ymin=-2
ymax=240
# ymax=150
plt.ylim(ymin, ymax)
# plt.yticks(range(0, ymax, 20), fontsize=font_size)
plt.yticks(range(0, ymax, 40), fontsize=font_size)

plt.ylabel('Occupied Cells Observed', fontsize=font_size)
plt.xlabel('Number of Robots', fontsize=font_size)

plt.subplots_adjust(bottom=0.25, top=0.9)

# Planning categories
# x = 2.3
# plt.plot([x, x], [ymin, ymax], '--', color='#AAAAFF')
# x = 6.3
# plt.plot([x, x], [ymin, ymax], '--', color='#AAAAFF')
# x = 10.3
# plt.plot([x, x], [ymin, ymax], '--', color='#AAAAFF')
# x = 14.3
# plt.plot([x, x], [ymin, ymax], '--', color='#AAAAFF')
# plt.text(3, 0.5, 'Proposed CNN\nReward Function',ha='center',color='#0000DD', fontsize=font_size)

# x = 16.5
# plt.plot([x, x], [ymin, ymax], '--', color='#AAAAFF')
# plt.text(12, 0.5, 'Cell-Counting\nReward Function',ha='center',color='#0000DD', fontsize=font_size)

plt.grid(axis='y', linestyle=':', linewidth=1, color='#CCCCCC')


# Ship it!
plt.show(block=True)



 
