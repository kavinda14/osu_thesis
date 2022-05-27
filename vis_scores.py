# Adapted from https://www.geeksforgeeks.org/how-to-create-boxplots-by-group-in-matplotlib/
from cmath import pi
import matplotlib.pyplot as plt
import numpy as np
import pickle
from utils import get_CONF, get_json_comp_conf
import os

# get the file paths from the .json config
CONF = get_CONF()
json_comp_conf = get_json_comp_conf()

# get the length of shared_files folder
_, _, files = next(os.walk(CONF[json_comp_conf]["shared_files_path"]))
file_count = len(files)

# this holds all the pickle data
pickles_dict = dict()
full_comms_planners = ["MCTS_network_network_fullnet", "MCTS_random_network_full",
                       "CellCountPlanner_fullnet", "MCTS_cellcount_cellcount_full", "MCTS_random_cellcount_full", "CellCountPlanner_full", "RandomPlanner_full"]
partial_comms_planners = ["MCTS_network_network_partialnet", "MCTS_random_network_partial",
                          "CellCountPlanner_partialnet", "MCTS_cellcount_cellcount_partial", "MCTS_random_cellcount_partial", "CellCountPlanner_partial", "RandomPlanner_partial"]
poor_comms_planners = ["MCTS_network_network_poornet", "MCTS_random_network_poor",
                       "CellCountPlanner_poornet", "MCTS_cellcount_cellcount_poor", "MCTS_random_cellcount_poor", "CellCountPlanner_poor", "RandomPlanner_poor"]
all_planners = full_comms_planners + partial_comms_planners + poor_comms_planners

for planner in all_planners:
    pickles_dict[planner] = list()    

for i in range(file_count):
    filename = CONF[json_comp_conf]["shared_files_path"] +  "scores_r4_t50_s20_{}".format(i+1)
    infile = open(filename, 'rb')
    temp_dict = pickle.load(infile)

    # append values for each planner
    for key in temp_dict.keys():
        if key in pickles_dict:
            pickles_dict[key] += temp_dict[key]

    infile.close()


# combine the pickles_dict planners in the dicts given above
full_comms_dict = {k: list() for k in full_comms_planners}
partial_comms_dict = {k: list() for k in partial_comms_planners}
poor_comms_dict = {k: list() for k in poor_comms_planners}

for key in full_comms_dict.keys():
    if key in pickles_dict.keys():
        full_comms_dict[key] += pickles_dict[key]
for key in partial_comms_dict.keys():
    if key in pickles_dict.keys():
        partial_comms_dict[key] += pickles_dict[key]
for key in poor_comms_dict.keys():
    if key in pickles_dict.keys():
        poor_comms_dict[key] += pickles_dict[key]


# create the score lists for the boxplot                
results_full_comms = [np.array(value) for value in full_comms_dict.values()]
results_partial_comms = [np.array(value) for value in partial_comms_dict.values()]
results_poor_comms = [np.array(value) for value in poor_comms_dict.values()]

ticks = ['MCTS (Greedy)', 'MCTS (Random)', 'Greedy', 'MCTS (Greedy)', 'MCTS (Random)', 'Greedy', 'Random']
# Do the plots
full_comms_plot = plt.boxplot(results_full_comms, 
                positions=np.array(np.arange(len(results_full_comms)))*3.0-0.6,widths=0.5, patch_artist=True,)
partial_comms_plot = plt.boxplot(results_partial_comms,
                positions=np.array(np.arange(len(results_partial_comms)))*3.0+0,widths=0.5, patch_artist=True,)
poor_comms_plot = plt.boxplot(results_poor_comms,
                positions=np.array(np.arange(len(results_poor_comms)))*3.0+0.6,widths=0.5, patch_artist=True,)

# Do formatting
def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color='k')
    for b in plot_name['boxes']:
    	b.set_facecolor(color_code)
         
    # use plot function to draw a small line to name the legend.
    plt.plot([], 's', c='w',mec='k', mfc=color_code, label=label)
    plt.legend(loc='upper right')
 
 
# Setting colors for each groups
define_box_properties(full_comms_plot, '#55FF22', 'Full Communication')
define_box_properties(partial_comms_plot, '#22BBFF', 'Partial Communication')
define_box_properties(poor_comms_plot, '#FF5555', 'Poor Communication')
 
# Other formatting
plt.xticks(np.arange(0, len(ticks) * 3, 3), ticks, rotation=40, ha='right', rotation_mode='anchor')
plt.xlim(-1.5, len(ticks)*3-1.5)
 
ymin=-2
ymax=130
plt.ylim(ymin, ymax)
plt.yticks(range(0,ymax,10))

plt.ylabel('Cells Observed')
plt.xlabel('Planning Algorithm')

plt.subplots_adjust(bottom=0.25, top=0.9)

# Planning categories
x = 7.5
plt.plot([x, x], [ymin, ymax], '--', color='#AAAAFF')
plt.text(3, 0.5, 'Proposed CNN\nReward Function',ha='center',color='#0000DD')

x = 16.5
plt.plot([x, x], [ymin, ymax], '--', color='#AAAAFF')
plt.text(12, 0.5, 'Cell-Counting\nReward Function',ha='center',color='#0000DD')

plt.grid(axis='y', linestyle=':', linewidth=1, color='#CCCCCC')
 
# Ship it!
plt.show(block=True)


 
