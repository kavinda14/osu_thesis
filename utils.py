import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import randint
import json
import math

# json conf for working with diff directories from diff computers
def get_CONF():
    with open("paths_conf.json") as json_conf:
        return json.load(json_conf)

def get_json_comp_conf():
    # json_comp_conf = "graeme desktop"
    # json_comp_conf = "macbook - kavi"
    json_comp_conf = "alienware - kavi"
    return json_comp_conf

# used to create random, valid starting locs
def get_random_loc(belief_map):
    valid_start_loc = False
    bounds = belief_map.get_bounds()
    while not valid_start_loc:
        x = randint(0, bounds[0]-1)
        y = randint(0, bounds[0]-1)
        valid_start_loc = belief_map.is_valid_loc([x, y])
    return [x, y]

def euclidean_distance(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    return math.sqrt((y2-y1)**2 + (x2-x1)**2)

def oracle_visualize(robots, bounds, map, planner, reward_type=None, rollout_type=None):
    plt.xlim(0, bounds[0])
    plt.ylim(0, bounds[1])
    # plt.title("Planner: {}, Score: {}".format(self.planner, sum(self.sensor_model.get_final_scores())))

    ax = plt.gca()
    ax.set_aspect('equal', 'box')

    # this has to be done before the bot for loop to avoid red patches
    # ..going over the other obs_occupied patches
    for spot in map.unobs_occupied:
        hole = patches.Rectangle(spot, 1, 1, facecolor='red')
        ax.add_patch(hole)

    for spot in map.unobs_free:
        hole = patches.Rectangle(spot, 1, 1, facecolor='black')
        ax.add_patch(hole)

    # get all the observed locations from all robots
    obs_free = set()
    obs_occupied = set()
    for bot in robots:
        bot_map = bot.get_map()

        obs_free = obs_free.union(bot_map.get_obs_free())
        obs_occupied = obs_occupied.union(bot_map.get_obs_occupied())

        bot_color = bot.get_color()

        # plot robot
        robot_x = bot.get_loc()[0] + 0.5
        robot_y = bot.get_loc()[1] + 0.5
        plt.scatter(robot_x, robot_y, color=bot_color, zorder=5)

        # plot robot path
        x_values = list()
        y_values = list()
        for path in bot.get_sensor_model().get_final_path():
            x_values.append(path[0] + 0.5)
            y_values.append(path[1] + 0.5)
        plt.plot(x_values, y_values, color=bot_color)

        for spot in obs_occupied:
            hole = patches.Rectangle(spot, 1, 1, facecolor=bot_color)
            ax.add_patch(hole)
        obs_occupied = set()

    for spot in obs_free:
        hole = patches.Rectangle(spot, 1, 1, facecolor='white')
        ax.add_patch(hole)

    if reward_type and rollout_type is not None:
        plt.title(planner + "_" + rollout_type + "_" + reward_type)
    else:
        plt.title(planner)

    plt.show()


## MCTS classes + functions ##

class State():
    def __init__(self, action, location):
        self.action = action
        self.location = location

        self.id = -1

        if self.action == 'left':
            self.id = 0
        elif self.action == 'right':
            self.id = 1
        elif self.action == 'forward':
            self.id = 2
        elif self.action == 'backward':
            self.id = 3

    def get_action(self):
        return self.action

    def get_location(self):
        return self.location

# returns valid State objects (contains action and location) from a given position


def generate_valid_neighbors(current_state, state_sequence, robot):
    neighbors = list()
    current_loc = current_state.get_location()

    sequence = [state.get_location() for state in state_sequence]
    actions = ['left', 'right', 'forward', 'backward']
    for action in actions:
        valid, new_loc = robot.check_valid_move_mcts(action, current_loc, True)
        if valid and new_loc not in sequence:
            neighbors.append(State(action, new_loc))

    # condition added because rollout_random ends up in spot with no neighbors sometimes
    if len(neighbors) == 0:
        while True:
            action_idx = randint(0, len(actions)-1)
            action = actions[action_idx]
            new_loc = robot.get_action_loc(action, curr_loc=current_loc)
            if robot.check_new_loc(new_loc[0], new_loc[1]):
                neighbors.append(State(action, new_loc))
                break

    return neighbors
