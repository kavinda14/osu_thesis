import NeuralNet
import pickle
from utils import get_CONF, get_json_comp_conf
import torch


if __name__ == "__main__":

    CONF = get_CONF()
    json_comp_conf = get_json_comp_conf()

    # unpickle all the data
    print("Unpickling started!")
    # filename = CONF[json_comp_conf]["pickle_path"] + "data_41x41_depoeharbor_cellcount_r4_t400_s80_rollout:True"
    filename = CONF[json_comp_conf]["pickle_path"] + "data_41x41_circular_oracle_r4_t1100_s20_rollout:True"
    infile = open(filename,'rb')
    data = pickle.load(infile)
    infile.close()
    print("Unpickling done!")


    # train network - initial 
    epochs = 1
    # this is the path where the NN weights will be saved
    # weights_path = CONF[json_comp_conf]["neural_net_weights_path"] + "depoeharbor_41x41_epoch{}_oracle_r4_t400_s80_rollout:True_batch128".format(epochs)
    weights_path = CONF[json_comp_conf]["neural_net_weights_path"] + "circular_21x21_epoch{}_oracle_r4_t1100_s20_rollout:True_batch128".format(epochs)
    print("Training network")
    NeuralNet.train_net(data, epochs, weights_path)
    
    # train already trained network
    # weights_path = "/home/kavi/thesis/neural_net_weights/circles_21x21_epoch3_random_oraclecellcount_r4_t1500_s35_rollout:False_samestartloc_batch128_actionindic"
    # weight_file = "/home/kavi/thesis/neural_net_weights/circles_21x21_epoch3+1_random_oraclecellcount_r4_t1500_s35_rollout:False_samestartloc_batch128_actionindic_retrain1"
    # neural_model = NeuralNet.Net(bounds)
    # neural_model.load_state_dict(torch.load(weights_path))
    # neural_model.cuda() # moves model to GPU
    # NeuralNet.train_net(data, bounds, epochs, weight_file, net=neural_model)



