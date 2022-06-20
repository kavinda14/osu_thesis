

from basic_MCTS_python import mcts
import random
import NeuralNet
import sys
sys.path.insert(0, './basic_MCTS_python')


def backtrack_count(exec_path, comm_exec_path, potential_loc):
    return exec_path.count(potential_loc) + comm_exec_path.count(potential_loc)


def random_planner(bot, sys_actions):
    bot_belief_map = bot.get_belief_map()
    curr_bot_loc = bot.get_loc()
    bot_exec_path = bot.get_exec_path()
    bot_comm_exec_path = bot.get_comm_exec_path()

    counter = 0  # incase it gets stuck in while loop
    while True:
        counter += 1
        action = random.choice(sys_actions)
        visited_before = True  # check if the loc has been visited before
        valid_move = bot_belief_map.is_valid_action(action, curr_bot_loc)
        potential_loc = bot_belief_map.get_action_loc(action, curr_bot_loc)

        if backtrack_count(bot_exec_path, bot_comm_exec_path, potential_loc) <= 1:
            visited_before = False

        if (valid_move and not visited_before) or (counter > 10):
            break

    return action


# model here is the neural net
def cellcount_planner(sys_actions, bot, sensor_model, neural_model, device, oracle=False, ground_truth_map=None, robot_occupied_locs=None, sense_range=None):
    best_action = random.choice(sys_actions)
    best_action_score = float('-inf')
    bot_exec_paths = bot.get_exec_path()
    bot_comm_exec_paths = bot.get_comm_exec_path()
    bot_belief_map = bot.get_belief_map()
    bot_sense_range = bot.get_sense_range()
    curr_bot_loc = bot.get_loc()

    # create map and path matrices for network 
    if neural_model is not None:
        partial_info = [sensor_model.create_partial_info(False)]
        partial_info_binary_matrices = sensor_model.create_binary_matrices(partial_info)
        path_matrix = sensor_model.create_path_matrix(False)

    for action in sys_actions:
        if bot_belief_map.is_valid_action(action, curr_bot_loc):
            potential_loc = bot_belief_map.get_action_loc(action, curr_bot_loc) # tuple is needed here for count()
            
            if backtrack_count(bot_exec_paths, bot_comm_exec_paths, potential_loc) <= 1:
                if neural_model is not None:
                    # we put partial_info and final_actions in a list because that's how those functions needed them in SensorModel
                    action_matrix = [sensor_model.create_action_matrix(action, curr_bot_loc, True)]
                    action_binary_matrices = sensor_model.create_binary_matrices(action_matrix)
                    
                    input = NeuralNet.create_image(partial_info_binary_matrices, path_matrix, action_binary_matrices)

                    # the unsqueeze adds an extra dimension at index 0 and the .float() is needed otherwise PyTorch will complain
                    # by unsqeezing, we add a batch dimension to the input, which is required by PyTorch: (n_samples, channels, height, width) 
                    input = input.unsqueeze(0).float().to(device)

                    action_score = neural_model(input).item()
                    
                else:
                    action_score = len(bot_belief_map.count_unknown_cells(bot_sense_range, potential_loc))
                    if oracle: # we use ground truth to get the observed locs
                        occupied_locs = ground_truth_map.get_observation(potential_loc, sense_range)[0]
                        
                        for loc in occupied_locs:
                            if loc not in robot_occupied_locs:
                                action_score += 1
                # if oracle:
                #     if action_score > best_action_score:
                #         best_action_score = action_score
                #         best_action = action
                # else:
                if action_score > best_action_score:
                    best_action_score = action_score
                    best_action = action
    
    return best_action


class Planner:
    def __init__(self, comm_step, comm_type):
        self.comm_step = comm_step
        self.sys_actions = ['left', 'right', 'backward', 'forward']
        self.name = "{}_{}".format(self.__class__.__name__, comm_type)

    def get_comm_step(self):
        return self.comm_step

    def get_sys_actions(self):
        return self.sys_actions

    def get_name(self):
        return self.name

class RandomPlanner(Planner):
    def __init__(self, comm_step, comm_type):
        super().__init__(comm_step, comm_type)
        self.comm_step = comm_step

    def get_action(self, bot):
        return random_planner(bot, self.get_sys_actions())

class CellCountPlanner(Planner):
    def __init__(self, neural_model, device, comm_step, comm_type):
        super().__init__(comm_step, comm_type)
        self.comm_step = comm_step
        self.neural_model = neural_model
        self.device = device

    def get_action(self, bot):
        return cellcount_planner(self.get_sys_actions(), bot, bot.get_sensor_model(), self.neural_model, self.device)

class OracleCellCountPlanner(Planner):
    def __init__(self, sense_range, neural_model, device, comm_step, comm_type):
        super().__init__(comm_step, comm_type)
        self.comm_step = comm_step
        self.sense_range = sense_range # added here because it is used in get_observation() in GroundTruthMap
        self.neural_model = neural_model
        self.device = device
        self.ground_truth_map = None
        self.robot_occupied_locs = set()

    def get_action(self, bot):
        return cellcount_planner(self.get_sys_actions(), bot, bot.get_sensor_model(), self.neural_model, self.device, True, self.ground_truth_map, self.robot_occupied_locs, self.sense_range)

    def set_ground_truth_map(self, map):
        self.ground_truth_map = map

    def set_robot_occupied_locs(self, locs):
        self.robot_occupied_locs = locs 

class MCTS(Planner):
    def __init__(self, rollout, reward, comm_step, comm_type, neural_model, device):
        super().__init__(comm_step, comm_type)
        self.rollout = rollout
        self.reward = reward
        self.budget = 6
        self.max_iter = 1000
        if self.reward == "network":
            # self.explore_exploit_param = 18.0 # =1.0 is recommended. <1.0 more exploitation. >1.0 more exploration.
            self.explore_exploit_param = 10.0 # =1.0 is recommended. <1.0 more exploitation. >1.0 more exploration.
            # self.explore_exploit_param = 4.0 
        else:
            # self.explore_exploit_param = 11.0  
            self.explore_exploit_param = 45.0  

        self.comm_step = comm_step
        self.comm_type = comm_type
        
        self.neural_model = neural_model
        self.device = device

        self.name = "{}_{}_{}_{}".format(self.__class__.__name__, self.rollout, self.reward, comm_type)
      
    def get_action(self, bot):
        # at curr_step == 0, nothing is explored so the generate_valid_neighbors() will be in an infinite loop because..
        # .. is_valid_action() always returns False
        # so at the very first step we can run cellcount or random planner so that the bot has free_locs to iterate over in is_valid_action()
        if len(bot.get_exec_path()) == 0:
            return cellcount_planner(self.get_sys_actions(), bot, bot.get_sensor_model(), self.neural_model, self.device)

        return mcts.mcts(self.budget, self.max_iter, self.explore_exploit_param, 
                         bot, self.rollout, self.reward, self.neural_model, self.device)
    
    def get_name(self):
        return self.name
