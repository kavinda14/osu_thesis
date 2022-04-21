from msilib.schema import Class
import random
import NeuralNet
from utils import backtrack_count


def random_planner(bot, robot_curr_locs, actions):
    bot_belief_map = bot.get_belief_map()
    curr_bot_loc = bot.get_loc()

    counter = 0  # incase it gets infinitely stuck in while loop
    while True:
        counter += 1
        action = random.choice(actions)
        visited_before = True  # check if the loc has been visited before
        valid_move = bot_belief_map.is_valid_action(action, curr_bot_loc)
        potential_loc = bot_belief_map.get_action_loc(action, curr_bot_loc)

        exec_paths = bot.get_exec_paths()
        comm_exec_paths = bot.get_comm_exec_paths()

        if backtrack_count(exec_paths, comm_exec_paths, potential_loc) <= 1 \
                and (potential_loc not in robot_curr_locs):
            visited_before = False
        if ((valid_move == True) and (visited_before == False)) or (counter > 10):
            break

    return action

def backtrack_count(exec_path, comm_exec_path, potential_loc):
    return exec_path.count(potential_loc) + comm_exec_path.count(potential_loc)

class RandomFullComm:
    def __init__(self, actions, robots, fullcomm_step):
        self.fullcomm_step = fullcomm_step
        self.actions = actions
        self.robots = robots

    def run(self, bot, robot_curr_locs, step_count):
        random_planner(bot, robot_curr_locs, self.actions)
        if (step_count % self.fullcomm_step) == 0:
            bot.communicate_belief_map(bot, self.robots)

class RandomPartialComm:
    def __init__(self, actions, robots, partialcomm_step):
        self.partialcomm_step = partialcomm_step
        self.actions = actions
        self.robots = robots

    def run(self, bot, robot_curr_locs, step_count):
        random_planner(bot, robot_curr_locs, self.actions)
        if (step_count % self.partialcomm_step) == 0:
            bot.communicate_belief_map(bot, self.robots)

class RandomPoorComm:
    def __init__(self, actions, robots, poorcomm_step):
        self.poorcomm_step = poorcomm_step
        self.actions = actions
        self.robots = robots

    def run(self, bot, robot_curr_locs, step_count):
        random_planner(bot, robot_curr_locs, self.actions)
        if (step_count % self.poorcomm_step) == 0:
            bot.communicate_belief_map(bot, self.robots)

# model here is the neural net
def cellcount_planner(actions, bot, sensor_model, neural_model, robot_curr_locs, neural_net=False, device=False):
    best_action_score = float('-inf')
    best_action = random.choice(actions)
    exec_paths = bot.get_exec_paths()
    comm_exec_paths = bot.get_comm_exec_paths()

    partial_info = [sensor_model.create_partial_info(False)]
    partial_info_binary_matrices = sensor_model.create_binary_matrices(partial_info)
    path_matrix = sensor_model.create_final_path_matrix(False)

    for action in actions:
        if bot.check_valid_move(action):
            potential_loc = tuple(bot.get_action_loc(action)) # tuple is needed here for count()
            
            # backtrack possibility
            if backtrack_count(exec_paths, comm_exec_paths, potential_loc) <= 1 \
                and (potential_loc not in robot_curr_locs):
                if neural_net:
                    # We put partial_info and final_actions in a list because that's how those functions needed them in SensorModel
                    final_actions = [sensor_model.create_action_matrix(action, True)]
                    final_actions_binary_matrices = sensor_model.create_binary_matrices(final_actions)
                
                    input = NeuralNet.create_image(partial_info_binary_matrices, path_matrix, final_actions_binary_matrices)

                    # The unsqueeze adds an extra dimension at index 0 and the .float() is needed otherwise PyTorch will complain
                    # By unsqeezing, we add a batch dimension to the input, which is required by PyTorch: (n_samples, channels, height, width) 
                    input = input.unsqueeze(0).float().to(device)

                    action_score = neural_model(input).item()
                    
                else:
                    action_score = len(bot.count_unknown_cells())

                if action_score > best_action_score:
                    best_action_score = action_score
                    best_action = action

    robot_curr_locs.add(tuple(bot.get_action_loc(best_action)))
    return best_action

class CellCountFullComm:
    def __init__(self, actions, robots, neural_model, fullcomm_step):
        self.fullcomm_step = fullcomm_step
        self.actions = actions
        self.robots = robots
        self.neural_model = neural_model

    def run(self, bot, robot_curr_locs, step_count):
        cellcount_planner(self.actions, bot, bot.get_sensor_model(), self.neural_model, robot_curr_locs)
        if (step_count % self.fullcomm_step) == 0:
            bot.communicate_belief_map(bot, self.robots)


class CellCountPartialComm:
    def __init__(self, actions, robots, neural_model, partialcomm_step):
        self.partialcomm_step = partialcomm_step
        self.actions = actions
        self.robots = robots
        self.neural_model = neural_model

    def run(self, bot, robot_curr_locs, step_count):
        cellcount_planner(self.actions, bot, bot.get_sensor_model(), self.neural_model, robot_curr_locs)
        if (step_count % self.partialcomm_step) == 0:
            bot.communicate_belief_map(bot, self.robots)


class CellCountPoorComm:
    def __init__(self, actions, robots, neural_model, poorcomm_step):
        self.poorcomm_step = poorcomm_step
        self.actions = actions
        self.robots = robots
        self.neural_model = neural_model

    def run(self, bot, robot_curr_locs, step_count):
        cellcount_planner(self.actions, bot, bot.get_sensor_model(), self.neural_model, robot_curr_locs)
        if (step_count % self.poorcomm_step) == 0:
            bot.communicate_belief_map(bot, self.robots)

        




