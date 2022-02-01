import NeuralNet
import pickle
import torch

if __name__ == "__main__":

    # unpickle all the data
    print("Unpickling started!")
    # alienware
    # filename = '/home/kavi/thesis/pickles/data_21x21_circles_random_greedyo_r4_t1000_s50_norollout_diffstartloc'
    # filename = '/home/kavi/thesis/pickles/data_21x21_circles_random_greedyo_r4_t1000_s50_rollout_diffstartloc'
    filename = '/home/kavi/thesis/pickles/data_21x21_circles_random_greedyno_r4_t2000_s25_rollout_diffstartloc'
    # macbook
    # filename = '/Users/kavisen/osu_thesis/data/data_21x21_circles_random_greedyno_r4_t800_s50_rollout'
    infile = open(filename,'rb')
    data = pickle.load(infile)
    infile.close()
    print("Unpickling done!")

    # this is the path where the NN weights will be saved
    
    # alienware
    # weights_path = "/home/kavi/thesis/neural_net_weights/circles_21x21_epoch4_random_greedyo_r4_t1000_s50_norollout_diffstartloc"
    weights_path = "/home/kavi/thesis/neural_net_weights/circles_21x21_epoch1_random_greedyo_r4_t2000_s25_rollout_diffstartloc"
    # macbookcircles_random_greedyo_r4_t2000_s25_rollout_diffstartloc
    # weights_path = "/Users/kavisen/osu_thesis/neural_net_weights/circles_21x21_epoch5_random_greedyno_r4_t800_s50_rollout"

    # train network - initial 
    bounds = [21, 21]
    epochs = 1
    print("Training network")
    # NeuralNet.run_network(data, bounds, epochs, weights_path)
    
    # train already trained network
    weight_file = "/home/kavi/thesis/neural_net_weights/circles_21x21_epoch1_random_greedyo_r4_t2000_s25_rollout_diffstartloc_iter2"
    neural_model = NeuralNet.Net(bounds)
    neural_model.load_state_dict(torch.load(weights_path)) 
    NeuralNet.run_network(data, bounds, epochs, weight_file, net=neural_model)



