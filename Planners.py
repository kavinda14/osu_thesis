import random
import NeuralNet

def random_planner(bot, robot_curr_locs, actions):
    bot_belief_map = bot.get_belief_map()
    curr_bot_loc = bot.get_loc()
    bot_exec_path = bot.get_exec_path()
    bot_comm_exec_path = bot.get_comm_exec_path()

    counter = 0  # incase it gets stuck in while loop
    while True:
        counter += 1
        action = random.choice(actions)
        visited_before = True  # check if the loc has been visited before
        valid_move = bot_belief_map.is_valid_action(action, curr_bot_loc)
        potential_loc = bot_belief_map.get_action_loc(action, curr_bot_loc)

        if backtrack_count(bot_exec_path, bot_comm_exec_path, potential_loc) <= 1 \
                and (potential_loc not in robot_curr_locs):
            visited_before = False

        if (valid_move and not visited_before) or (counter > 10):
            break

    return action

def backtrack_count(exec_path, comm_exec_path, potential_loc):
    return exec_path.count(potential_loc) + comm_exec_path.count(potential_loc)


# model here is the neural net
def cellcount_planner(actions, bot, sensor_model, neural_model, robot_curr_locs, neural_net=False, device=False):
    best_action_score = float('-inf')
    best_action = random.choice(actions)
    exec_paths = bot.get_exec_path()
    comm_exec_paths = bot.get_comm_exec_path()
    bot_belief_map = bot.get_belief_map()
    curr_bot_loc = bot.get_loc()

    partial_info = [sensor_model.create_partial_info(False)]
    partial_info_binary_matrices = sensor_model.create_binary_matrices(partial_info)
    path_matrix = sensor_model.create_path_matrix(False)

    for action in actions:
        if bot_belief_map.is_valid_action(action, curr_bot_loc):
            potential_loc = bot_belief_map.get_action_loc(action, curr_bot_loc) # tuple is needed here for count()
            
            # backtrack possibility
            if backtrack_count(exec_paths, comm_exec_paths, potential_loc) <= 1 \
                and (potential_loc not in robot_curr_locs):
                if neural_net:
                    # We put partial_info and final_actions in a list because that's how those functions needed them in SensorModel
                    action_matrix = [sensor_model.create_action_matrix(action, True)]
                    action_binary_matrices = sensor_model.create_binary_matrices(action_matrix)
                
                    input = NeuralNet.create_image(partial_info_binary_matrices, path_matrix, action_binary_matrices)

                    # The unsqueeze adds an extra dimension at index 0 and the .float() is needed otherwise PyTorch will complain
                    # By unsqeezing, we add a batch dimension to the input, which is required by PyTorch: (n_samples, channels, height, width) 
                    input = input.unsqueeze(0).float().to(device)

                    action_score = neural_model(input).item()
                    
                else:
                    action_score = len(bot_belief_map.count_unknown_cells(bot.get_sense_range(), curr_bot_loc))

                if action_score > best_action_score:
                    best_action_score = action_score
                    best_action = action

    return best_action


class Planner:
    def __init__(self, comm_step):
        self.comm_step = comm_step

    def get_comm_step(self):
        return self.comm_step


class RandomPlanner(Planner):
    def __init__(self, actions, comm_step):
        super().__init__(comm_step)
        self.comm_step = comm_step
        self.actions = actions

    def get_action(self, bot, robot_curr_locs):
        return random_planner(bot, robot_curr_locs, self.actions)

class CellCountPlanner(Planner):
    def __init__(self, actions, neural_model, comm_step):
        super().__init__(comm_step)
        self.comm_step = comm_step
        self.actions = actions
        self.neural_model = neural_model

    def get_action(self, bot, robot_curr_locs):
        return cellcount_planner(self.actions, bot, bot.get_sensor_model(), self.neural_model, robot_curr_locs)



