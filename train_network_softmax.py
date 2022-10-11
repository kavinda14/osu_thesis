import NeuralNetSoftmax
import pickle
from utils import get_CONF, get_json_comp_conf
import torch


if __name__ == "__main__":

    CONF = get_CONF()
    json_comp_conf = get_json_comp_conf()

    # unpickle all the data
    print("Unpickling started!")
    # filename = CONF[json_comp_conf]["pickle_path"] + "data_41x41_depoeharbor_cellcount_r4_t400_s80_rollout:True"
    filename = CONF[json_comp_conf]["pickle_path"] + "data_41x41_depoeharbor_oracle_r4_t700_s50_rollout:False"
    infile = open(filename,'rb')
    data = pickle.load(infile)
    infile.close()
    print("Unpickling done!")


    # train network - initial 
    epochs = 2
    # this is the path where the NN weights will be saved
    weights_path = CONF[json_comp_conf]["neural_net_weights_path"] + "depoeharbor_41x41_epoch{}_oracle_r4_t700_s50_rollout:False_batch128".format(epochs)
    # weights_path = CONF[json_comp_conf]["neural_net_weights_path"] + "21x21_forest_oracle_r4_t4400_s20_epoch2_rollout:False"
    print("Training network")
    NeuralNetSoftmax.train_net(data, epochs, weights_path)
    
    # train already trained network
    # weights_path = "/home/kavi/thesis/neural_net_weights/data_41x41_depoeharbor_oracle_r4_t600_s50_rollout:False_softmax"
    # weight_file = "/home/kavi/thesis/neural_net_weights/data_41x41_depoeharbor_oracle_r4_t600_s50_rollout:False_softmax_retrain1"
    # neural_model = NeuralNetSoftmax.Net()
    # neural_model.load_state_dict(torch.load(weights_path))
    # neural_model.cuda() # moves model to GPU
    # NeuralNetSoftmax.train_net(data, epochs, weight_file, net=neural_model)



