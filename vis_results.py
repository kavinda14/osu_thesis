# Adapted from https://www.geeksforgeeks.org/how-to-create-boxplots-by-group-in-matplotlib/
import matplotlib.pyplot as plt
import numpy as np
import pickle
from util import get_CONF, get_json_comp_conf
import os

# get the file paths from the .json config
CONF = get_CONF()
json_comp_conf = get_json_comp_conf()

# get the length of shared_files folder
_, _, files = next(os.walk(CONF[json_comp_conf]["shared_files_path"]))
file_count = len(files)


# this holds all the pickle data
pickles_dict = dict()
for i in range(file_count):
    filename = CONF[json_comp_conf]["shared_files_path"] +  "trials100_steps15_allplanners_{}".format(i+1)
    infile = open(filename, 'rb')
    pickles_dict[i] = list(filter(None, pickle.load(infile))) # removes empty lists
    infile.close()

# list the planners in each list
full_comms_planners = ["net_fullcomm_net_fullcomm", "random_fullcomm_net_fullcomm",
                       "net_fullcomm", "greedy_fullcomm_greedy_fullcomm", "random_fullcomm_greedy_fullcomm", "greedy_fullcomm", "random_fullcomm"]
partial_comms_planners = ["net_partialcomm_net_partialcomm", "random_partialcomm_net_partialcomm",
                       "net_partialcomm", "greedy_partialcomm_greedy_partialcomm", "random_partialcomm_greedy_partialcomm", "greedy_partialcomm", "random_partialcomm"]
poor_comms_planners = ["net_poorcomm_net_poorcomm", "random_poorcomm_net_poorcomm",
                       "net_poorcomm", "greedy_poorcomm_greedy_poorcomm", "random_poorcomm_greedy_poorcomm", "greedy_poorcomm", "random_poorcomm"]

# aggregate the scores from the different pickles for the same planner
full_comms_dict = {k: list() for k in full_comms_planners}
partial_comms_dict = {k: list() for k in partial_comms_planners}
poor_comms_dict = {k: list() for k in poor_comms_planners}
for key, value in pickles_dict.items():
    for score_list in value:
        for key in full_comms_dict.keys():
            if score_list[0] == key:
                full_comms_dict[key] += score_list[1:]
        for key in partial_comms_dict.keys():
            if score_list[0] == key:
                partial_comms_dict[key] += score_list[1:]
        for key in poor_comms_dict.keys():
            if score_list[0] == key:
                poor_comms_dict[key] += score_list[1:]

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
ymax=90
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


 
