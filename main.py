from SensorModel import SensorModel
from Map import Map
from Robot import Robot
from Simulator import Simulator
import numpy as np

if __name__ == "__main__":

    bounds = [10, 10]
    map = Map(bounds, 1)
    robot = Robot(2, 2, bounds, map)
    sensor_model = SensorModel(robot, map)
    simulator = Simulator(map, robot, sensor_model)
    simulator.run(20, False)
    
    sensor_model.final_path_as_matrix()
    simulator.visualize()
    score = simulator.get_score()

    print("Score: ", score)

 
    