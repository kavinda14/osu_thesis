from SensorModel import SensorModel
from Map import Map
from Robot import Robot
from Simulator import Simulator
import NeuralNet
import random
import time
from tqdm import tqdm
import pickle

if __name__ == "__main__":

    # unpickle all the data
    print("Unpickling started!")
    filename = '/home/kavi/thesis/pickles/data_21x21_random_greedyo_t45_s2000'
    infile = open(filename,'rb')
    data = pickle.load(infile)
    infile.close()
    print("Unpickling done!")

    # this is the path where the NN weights will be saved
    weights_path = "/home/kavi/thesis/neural_net_weights/circles_random_21x21_epoch2_mctsrolloutdata3"
    
    # train network
    bounds = [21, 21]
    NeuralNet.runNetwork(data, bounds, weights_path)
    
   


