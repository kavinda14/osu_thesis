from SensorModel import SensorModel
from Map import Map
from Robot import Robot
from Simulator import Simulator
import NeuralNet

if __name__ == "__main__":

    input_partial_info_binary_matrices = list()
    input_path_matrices = list()
    input_actions_binary_matrices = list()
    input_scores = list()

    for _ in range(1): 
    # Bounds need to be an odd number for the action to always be in the middle
        bounds = [111, 111]
        map = Map(bounds, 400)
        robot = Robot(2, 2, bounds, map)
        sensor_model = SensorModel(robot, map)
        simulator = Simulator(map, robot, sensor_model)
        simulator.run(10, False)
        
        # Training data
        path_matricies = sensor_model.get_final_path_matrices()

        final_partial_info = sensor_model.get_final_partial_info()
        partial_info_binary_matrices = sensor_model.create_binary_matrices(final_partial_info)

        final_actions = sensor_model.get_final_actions()
        final_actions_binary_matrices = sensor_model.create_binary_matrices(final_actions)

        final_scores = sensor_model.get_final_scores()

        input_path_matrices = input_path_matrices + path_matricies
        input_partial_info_binary_matrices = input_partial_info_binary_matrices + partial_info_binary_matrices
        input_actions_binary_matrices = input_actions_binary_matrices + final_actions_binary_matrices
        input_scores = input_scores + final_scores

    print("final_path_matrices: ", len(input_path_matrices))
    print("final_partial_info_binary_matrices: ", len(input_partial_info_binary_matrices))
    print("final_final_actions_binary_matrices", len(input_actions_binary_matrices))
    print("final_final_scores: ", len(input_scores))
    
    # data = NeuralNet.datasetGenerator(partial_info_binary_matrices, path_matricies, final_actions_binary_matrices, final_scores)
    # data = NeuralNet.datasetGenerator(input_partial_info_binary_matrices, input_path_matrices, input_actions_binary_matrices, input_scores)
    # NeuralNet.runNetwork(data, bounds)

    # simulator.visualize()
    # score = simulator.get_score()

    # print("Score: ", score)

