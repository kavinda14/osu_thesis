import pickle
import matplotlib.pyplot as plt

# unpickle all the data
print("Unpickling started!")
# alienware
filename = '/home/kavi/thesis/pickles/data_21x21_circles_random_greedyo_r4_t1000_s50_norollout_diffstartloc'
# macbook
# filename = '/Users/kavisen/osu_thesis/data/data_21x21_circles_random_greedyno_r4_t800_s50_rollout'
infile = open(filename,'rb')
data = pickle.load(infile)
infile.close()
print("Unpickling done!")

# Box plot
score_lists_copy = score_lists
for score_list in score_lists_copy:
    score_list.remove(score_list[0])   

# do this otherwise x axis is not correct
for i in x_pos:
    x_pos[i] += 1

fig = plt.figure(figsize =(10, 7))
plt.boxplot(score_lists_copy)
plt.xticks(x_pos, bars, rotation=45)
plt.title(weight_file+"_trials:"+str(trials)+"_steps:"+str(steps))
plt.show()