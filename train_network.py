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
    # find better way to get this daeta from generate_data_circle.py
    filename = '/home/kavi/thesis/pickles/final_path_matrices_circle_21x21_random_greedyo_t45_s2000'
    infile = open(filename,'rb')
    input_path_matrices = pickle.load(infile)
    infile.close()

    filename = '/home/kavi/thesis/pickles/final_partial_info_binary_matrices_random_greedyo_t45_s2000'
    infile = open(filename,'rb')
    input_partial_info_binary_matrices = pickle.load(infile)
    infile.close()

    filename = '/home/kavi/thesis/pickles/final_final_actions_binary_matrices_circle_21x21_random_greedyo_t45_s2000'
    infile = open(filename,'rb')
    input_actions_binary_matrices = pickle.load(infile)
    infile.close()

    filename = '/home/kavi/thesis/pickles/final_final_scores_circle_21x21_random_greedyo_t45_s2000'
    infile = open(filename,'rb')
    input_scores = pickle.load(infile)
    infile.close()

    print("final_path_matrices: ", len(input_path_matrices))
    print("final_partial_info_binary_matrices: ", len(input_partial_info_binary_matrices))
    print("final_final_actions_binary_matrices", len(input_actions_binary_matrices))
    print("final_final_scores: ", len(input_scores))

    # this is the path where the NN weights will be saved
    weights_path = "/home/kavi/thesis/neural_net_weights/circles_random_21x21_epoch2_mctsrolloutdata3"
    bounds = [21, 21]

    # train network
    data = NeuralNet.datasetGenerator(input_partial_info_binary_matrices, input_path_matrices, input_actions_binary_matrices, input_scores)
    NeuralNet.runNetwork(data, bounds, weights_path)
    
   


