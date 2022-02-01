from operator import mod
import pickle
from typing import final
import matplotlib.pyplot as plt
import numpy as np

# Creates a box plot of the rewards of the generated data to see if there are varied rewards

# unpickle all the data
print("Unpickling started!")
# alienware
model = "data_21x21_circles_random_greedyo_r4_t2000_s25_rollout_diffstartloc"
filename = "/home/kavi/thesis/pickles/"+model
# macbook
# filename = '/Users/kavisen/osu_thesis/data/data_21x21_circles_random_greedyno_r4_t800_s50_rollout'
infile = open(filename,'rb')
data = pickle.load(infile)
infile.close()
print("Unpickling done!")

score_list = list()
zero_reward_count = 0
for i, _ in enumerate(data):
    reward = data[i][1]
    if reward == 0:
        zero_reward_count += 1
    score_list.append(data[i][1])

# print(score_list)
print("percentage of 0s: ", zero_reward_count/len(score_list))

max = np.amax(np.abs(score_list))
numpy_array = np.abs(score_list)
numpy_array_no_zeros = numpy_array[numpy_array!=0]
print(numpy_array_no_zeros)
# numpy_array_no_zeros = np.average(np.abs(score_list))
average = np.average(numpy_array_no_zeros)
print('max', max)
print('average', average)
normalized_score_list = np.asarray(score_list)*(1.0/average)
# print(normalized_score_list)

# Box plot
fig = plt.figure(figsize =(10, 7))
# plt.boxplot(score_list)
plt.boxplot(numpy_array_no_zeros)
plt.title(model)
plt.show()