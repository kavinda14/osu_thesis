from SensorModel import SensorModel
from Map import Map
from Robot import Robot
from Simulator import Simulator
import NeuralNet

if __name__ == "__main__":

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

    # print("len path: ", len(path_matricies))
    # print("path matrix: ", path_matricies[0])
    # print("len partial info: ", len(partial_info_binary_matrices[0]))
    # print("partial info :", partial_info_binary_matrices[0])
    # print("len actions: ", len(final_actions_binary_matrices))
    # print("actions: ",final_actions_binary_matrices[0] )
    # print("len score: ", len(final_scores))
    # print("score: ", final_scores[0])

    print(final_scores)
    
    # data = NeuralNet.datasetGenerator(partial_info_binary_matrices, path_matricies, final_actions_binary_matrices, final_scores)
    # NeuralNet.runNetwork(data, bounds)

    simulator.visualize()
    score = simulator.get_score()

    print("Score: ", score)

