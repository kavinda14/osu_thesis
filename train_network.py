import NeuralNet
import pickle
from utils import get_CONF, get_json_comp_conf


if __name__ == "__main__":

    CONF = get_CONF()
    json_comp_conf = get_json_comp_conf()

    # unpickle all the data
    print("Unpickling started!")
    filename = CONF[json_comp_conf]["pickle_path"] + "data_21x21_circles_random_cellcount_r4_t1500_s35_rollout:False_samestartloc_actionindic"
    infile = open(filename,'rb')
    data = pickle.load(infile)
    infile.close()
    print("Unpickling done!")


    # train network - initial 
    bounds = [21, 21]
    epochs = 4
    # this is the path where the NN weights will be saved
    weights_path = CONF[json_comp_conf]["neural_net_weights_path"] + "circles_21x21_epoch{}_random_oraclecellcount_r4_t1500_s35_rollout:False_samestartloc_batch128_actionindic".format(epochs)
    print("Training network")
    NeuralNet.run_network(data, bounds, epochs, weights_path)
    
    # train already trained network
    # weight_file = "/home/kavi/thesis/neural_net_weights/circles_21x21_epoch1_random_greedyo_r4_t2000_s25_rollout_diffstartloc_iter2"
    # neural_model = NeuralNet.Net(bounds)
    # neural_model.load_state_dict(torch.load(weights_path)) 
    # NeuralNet.run_network(data, bounds, epochs, weight_file, net=neural_model)



