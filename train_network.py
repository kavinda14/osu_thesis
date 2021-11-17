import NeuralNet
import pickle

if __name__ == "__main__":

    # unpickle all the data
    print("Unpickling started!")
    filename = '/home/kavi/thesis/pickles/data_21x21_circles_random_t600_s1000'
    infile = open(filename,'rb')
    data = pickle.load(infile)
    infile.close()
    print("Unpickling done!")

    # this is the path where the NN weights will be saved
    weights_path = "/home/kavi/thesis/neural_net_weights/circles_21x21_epoch10_random_t600_s1000"
    
    # train network
    bounds = [21, 21]
    epochs = 10
    NeuralNet.run_network(data, bounds, epochs, weights_path)
    
   


