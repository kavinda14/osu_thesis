from SensorModel import SensorModel
from Map import Map
from Robot import Robot
from Simulator import Simulator
import copy
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
 
    # Bounds need to be an odd number for the action to always be in the middle
    # planner_options = ["random", "greedy", "network"]
    planner_options = ["random"]
    # planner_options = ["random"]
    bounds = [41, 41]
    random = list()
    greedy = list()
    network = list()
    x1 = list()

    trials = 100
    for i in range(trials):
        print("Trial no: {}".format(i))
        x1.append(i)
        map = Map(bounds, 7, (), False)
        unobs_occupied = copy.deepcopy(map.get_unobs_occupied())
        for planner in planner_options:     
            map = Map(bounds, 18, copy.deepcopy(unobs_occupied), True)
            robot = Robot(0, 0, bounds, map)
            sensor_model = SensorModel(robot, map)
            simulator = Simulator(map, robot, sensor_model, planner)
            # simulator.visualize()
            simulator.run(7000, False)
            simulator.visualize()
            score = sum(sensor_model.get_final_scores())
            print("Planner: {}, Score: {}".format(planner, score))

            if planner == "random":
                random.append(score)
            elif planner == "greedy":
                greedy.append(score)
            else:
                network.append(score)

    avg_random = sum(random)/trials
    avg_greedy = sum(greedy)/trials
    avg_network = sum(network)/trials

    plt.plot(x1, random, label = "random")
    plt.plot(x1, greedy, label = "greedy")
    plt.plot(x1, network, label = "network")

    plt.xlabel('Trial no')
    # Set the y axis label of the current axis.
    plt.ylabel('Score')
    # Set a title of the current axes.
    plt.title('Avg scores: random: {}, greedy: {}, network: {}'.format(avg_random, avg_greedy, avg_network))
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()


