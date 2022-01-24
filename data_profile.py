from operator import mod
import pickle
from typing import final
import matplotlib.pyplot as plt

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
for i, _ in enumerate(data):
    score_list.append(data[i][1])

print(score_list)

# Box plot
fig = plt.figure(figsize =(10, 7))
plt.boxplot(score_list)
plt.title(model)
plt.show()