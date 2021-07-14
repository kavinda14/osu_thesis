import copy
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Simulator:
    def __init__(self, world_map, robot):
        """
        Inputs:
        world_map: The map to be explored
        robot: the Robot object from Robot.py
        """
        self.map = copy.deepcopy(world_map)
        self.robot = copy.deepcopy(robot)

        # The following identify what has been seen by the robot
        self.obs_occupied = set()
        self.obs_free = set()
        self.score = 0
        self.iterations = 0
    
    def run(self):
        duration = max(len(self.robot.final_path))
        for _ in range(0, duration):
            end = self.tick()
            if end:
                break

    def tick(self):
        self.iterations += 1

        # Update the location of the robots
        # Generate an action from the robot path
        action = self.robot.follow_path()
        # Move the robot
        self.robot.move(action)

        # Update the explored map based on robot position
        self._update_map()

        # Update the score
        self.score = len(self.obs_occupied)*10

        #End when all survivors have been reached OR 1,000 iterations
        if (len(self.obs_occupied) == self.map.unobs_occupied) or (self.iterations == 1000):
            self.found_goal = len(self.obs_occupied) == self.map.unobs_occupied
            return True
        else:
            return False

    def reset_game(self):
        self.iterations = 0
        self.score = 0
        self.obs_occupied = set()
        self.obs_free = set()

    def get_score(self):
        return self.score

    def get_iteration(self):
        return self.iterations

    def _update_map(self):
        # Sanity check the robot is in bounds
        if not self.robot.check_valid_loc():
            print(self.robot.get_loc())
            raise ValueError(f"Robot has left the map. It is at position: {self.robot.get_loc()}, outside of the map boundary")
        
        new_observations = self.map.scan(self.robot.get_loc(), self.robot.sensing_range)
        self.obs_occupied = self.obs_occupied.union(new_observations[0])
        self.obs_free = self.obs_free.union(new_observations[1])

    def visualize(self):
        plt.xlim(self.map.bounds[0]-.5, self.map.bounds[1]+(self.map.bounds[1]*.05))
        plt.ylim(self.map.bounds[0]-.5, self.map.bounds[1]+(self.map.bounds[1]*.05))
        ax = plt.gca()

        #obs_oc, obs_free, unobs_oc, unobs_free

        unobs_occupied_x = [i[0] for i in self.map.unobs_occupied]
        unobs_occupied_y = [i[1] for i in self.map.unobs_occupied]
        plt.scatter(unobs_occupied_x, unobs_occupied_y, color='tab:red')

        unobs_free_x = [i[0] for i in self.map.unobs_free]
        unobs_free_y = [i[1] for i in self.map.unobs_free]
        plt.scatter(unobs_free_x, unobs_free_y, color='tab:gray')

        hotspot_x = [i[0] for i in self.map.hotspots]
        hotspot_y = [i[1] for i in self.map.hotspots]
        plt.scatter(hotspot_x, hotspot_y, color='black', marker="x")

        for spot in self.map.unobs_occupied:
            hole = patches.Rectangle(spot, 1, 1, linewidth=2, facecolor='red')
            ax.add_patch(hole)

        for spot in self.map.unobs_free:
            hole = patches.Rectangle(spot, 1, 1, linewidth=2, facecolor='black')
            ax.add_patch(hole)
        
        for spot in self.obs_free:
            hole = patches.Rectangle(spot, 1, 1, linewidth=2, facecolor='white')
            ax.add_patch(hole)

        for spot in self.obs_occupied:
            hole = patches.Rectangle(spot, 1, 1, linewidth=2, facecolor='green')
            ax.add_patch(hole)

        robot_x = self.robot.get_loc()[0]
        robot_y = self.robot.get_loc()[1]
        # robot_x = [p.location[0] for p in r.final_path]
        # robot_y = [p.location[1] for p in r.final_path]
        plt.plot(robot_x, robot_y)

        plt.show()