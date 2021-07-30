from SensorModel import SensorModel
from Map import Map
from Robot import Robot
from Simulator import Simulator
import numpy as np

if __name__ == "__main__":

    bounds = [8, 8]
    map = Map(bounds, 2)
    robot = Robot(2, 2, bounds, map)
    sensor_model = SensorModel(robot, map)
    simulator = Simulator(map, robot, sensor_model)
    simulator.run(1, False)
    
    sensor_model.final_path_as_matrix()
    binary_matrices = sensor_model.final_partial_info_as_binary_matrices()
    print("final_partial_info: ", sensor_model.final_partial_info[0])
    print("binary matrix: ", binary_matrices[0])
    simulator.visualize()
    score = simulator.get_score()

    print("Score: ", score)

 