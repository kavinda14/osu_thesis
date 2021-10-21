from SensorModel import SensorModel
from Map import Map
from Robot import Robot
from Simulator import Simulator
import copy
import matplotlib.pyplot as plt
import random as r
import time as time
import sys
sys.path.insert(0, './basic_MCTS_python')
from basic_MCTS_python import mcts


if __name__ == "__main__":
 
    # Bounds need to be an odd number for the action to always be in the middle
    bounds = [21, 21]

    map = Map(bounds, 7, (), False)
    unobs_occupied = copy.deepcopy(map.get_unobs_occupied())

    valid_starting_loc = False
    while not valid_starting_loc:
        x = r.randint(0, bounds[0]-1)
        y = r.randint(0, bounds[0]-1)
        valid_starting_loc = map.check_loc(x, y) 

    map = Map(bounds, 18, copy.deepcopy(unobs_occupied), True)
    robot = Robot(x, y, bounds, map)
    
    sensor_model = SensorModel(robot, map)
    simulator = Simulator(map, robot, sensor_model, 'greedy')
    # simulator.visualize()

    # Setup the problem
    # num_actions = 3
    # action_set = []
    # for i in range(num_actions):
    #     id = i
    #     action_set.append(Action(id,i))
    budget = 7

    exploration_exploitation_parameter = 0.8 # =1.0 is recommended. <1.0 more exploitation. >1.0 more exploration. 
    max_iterations = 2000
    
    [solution, root, list_of_all_nodes, winner] = mcts.mcts(budget, max_iterations, exploration_exploitation_parameter, robot, map)
    print(winner)
  


