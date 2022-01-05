import NeuralNet
import pickle

if __name__ == "__main__":

    # unpickle all the data
    print("Unpickling started!")
    # alienware
    filename = '/home/kavi/thesis/pickles/data_21x21_circles_random_greedyno_r4_t800_s50_rollout'
    # macbook
    # filename = '/Users/kavisen/osu_thesis/data/data_21x21_circles_random_greedyno_r4_t800_s50_rollout'
    infile = open(filename,'rb')
    data = pickle.load(infile)
    infile.close()
    print("Unpickling done!")

    # this is the path where the NN weights will be saved
    
    # alienware
    weights_path = "/home/kavi/thesis/neural_net_weights/circles_21x21_epoch2_random_greedyno_r4_t800_s50_rollout"
    # macbook
    # weights_path = "/Users/kavisen/osu_thesis/neural_net_weights/circles_21x21_epoch5_random_greedyno_r4_t800_s50_rollout"

    # train network
    bounds = [21, 21]
    epochs = 2
    print("Training network")
    NeuralNet.run_network(data, bounds, epochs, weights_path)
    
   


