from SensorModel import SensorModel
from Map import Map
from Robot import Robot
from Simulator import Simulator

if __name__ == "__main__":
 
    # Bounds need to be an odd number for the action to always be in the middle
    bounds = [111, 111]
    map = Map(bounds, 400)
    robot = Robot(2, 2, bounds, map)
    sensor_model = SensorModel(robot, map)
    simulator = Simulator(map, robot, sensor_model, "network")
    simulator.run(100, False)
    
    simulator.visualize()
    score = sum(sensor_model.get_final_scores)
    print("Score: ", score)

   
