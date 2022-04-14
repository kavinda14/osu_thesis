import matplotlib.pyplot as plt
import pickle
import json
from utils import get_CONF, get_json_comp_conf

"""
This script is to see which step size is the most appropriate, where mcts performs the best.
"""

CONF = get_CONF()
json_comp_conf = get_json_comp_conf()

# unpickle scores
print("Unpickling started..")
filename = CONF[json_comp_conf]["pickle_path"] + "planner_scores_multibot/" + "trials1_steps70_accscore_5"
infile = open(filename, 'rb')
acc_score = pickle.load(infile)
infile.close()
print("Pickling done!")

print(acc_score)

# plot lines
x = [i for i in range(len(acc_score[0])-1)]
print(len(x))
for score in acc_score:
    plt.plot(x, score[1:], label=score[0])
plt.legend()
plt.show()
