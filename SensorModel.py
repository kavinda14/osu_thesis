import numpy as np

class SensorModel:
    def __init__(self, bot, belief_map):
        self.bot = bot
        self.belief_map = belief_map

        self.partial_info_matrices = list()
        self.path_matrices = list()
        self.comm_path_matrices = list()
        self.action_matrices = list()

    # Called in Simulator
    # We keep update as true for getting the training data
    def create_partial_info(self, update=True):
        bounds = self.belief_map.get_bounds()
        partial_info = np.empty((bounds[0], bounds[1]), dtype=int)

        unknown_locs = self.belief_map.get_unknown_locs()
        for loc in unknown_locs:
            partial_info[loc] = 2

        free_locs = self.belief_map.get_free_locs()
        for loc in free_locs:
            partial_info[loc] = 0

        occupied_locs = self.belief_map.get_occupied_locs()
        for loc in occupied_locs:
            partial_info[loc] = 1

        if update:
            self.partial_info_matrices.append(partial_info)
        else: 
            return partial_info

    
    # update flag was added because when running greedy planner with NN, we want to get path but not update final list
    def create_path_matrix(self, update=True, input_path=None):
        bounds = self.belief_map.get_bounds()
        path_matrix = np.zeros((bounds[0], bounds[1]), dtype=int)
        
        # input_path is for mcts rollout
        if input_path == None:
            exec_paths = self.bot.get_exec_path()
            for path in exec_paths:
                path_matrix[path] = 1
        else:
            for path in input_path:
                path_matrix[path] = 1
        
        # this is for multi-robot when communication of other_paths is done
        comm_exec_paths = self.bot.get_comm_exec_path()
        for path in comm_exec_paths:
            path_matrix[path] = 1

        if update:
            self.path_matrices.append(path_matrix)
        else:
            return path_matrix
    
    # this was created to combine path and others together for multi-robot rollout
    # it differs from create_path_matrix() because it adds ONLY the exec_paths and NOT comm_exec_paths
    def create_rollout_path_matrix(self, update=True):
        bounds = self.belief_map.get_bounds()
        path_matrix = np.zeros((bounds[0], bounds[1]), dtype=int)

        exec_paths = self.bot.get_exec_path()
        for path in exec_paths:
            path_matrix[path] = 1
        
        if update:
            self.path_matrices.append(path_matrix)

        else:
            return path_matrix

    # this was created to combine path and others together for multi-robot rollout
    def create_rollout_comm_path_matrix(self, update=True):
        bounds = self.belief_map.get_bounds()
        path_matrix = np.zeros((bounds[0], bounds[1]), dtype=int)

        comm_exec_paths = self.bot.get_comm_exec_path()
        for path in comm_exec_paths:
            path_matrix[path] = 1

        if update:
            self.comm_path_matrices.append(path_matrix)

        else:
            return path_matrix


    def create_binary_matrices(self, input_list):
        binary_matrices = list()
        
        for main_matrix in input_list:

            matrix_list = list()
            n = 1

            while (n <= 3):
                sub_matrix = np.empty((np.shape(main_matrix)), dtype=int)
                for x in range(np.shape(main_matrix)[0]):
                    for y in range(np.shape(main_matrix)[1]):
                        # free
                        if n == 1:
                            if main_matrix[x, y] == 0:
                                sub_matrix[x, y] = 1
                            else:
                                sub_matrix[x, y] = 0

                        # occupied
                        if n == 2:
                            if main_matrix[x, y] == 1:
                                sub_matrix[x, y] = 1
                            else:
                                sub_matrix[x, y] = 0

                        # unknown
                        if n == 3:
                            if main_matrix[x, y] == 2:
                                sub_matrix[x, y] = 1
                            else:
                                sub_matrix[x, y] = 0
                
                matrix_list.append(sub_matrix)
                n += 1

            binary_matrices.append(matrix_list)
        
        return binary_matrices

    # greedy_planner flag is added because we need to return action matrix
    # *args was added because in mcts network_reward, we need some way of passing the the robot location
    def create_action_matrix(self, action, curr_bot_loc, get_matrix=False):
        # Think of this as an action but a diff way of representing it
        # This function needs to be called before we move the robot in the Simulator

        # Create empty matrix of same bounds
        # Fill matrix in with the unknown digit = 1
        # Get the action location 
        # Get the mid-point of the matrix = [x/2, y/2]
        # Get displacement = mid-point - action_loc
        # Iterate through all the values of matrix1
            # If coord + displacement is within bounds:
                # matrix2[coord + displacement] = matrix1[coord]

        bounds = self.belief_map.get_bounds()
        action_matrix = np.full((bounds[0], bounds[1]), 2, dtype=int)
        mid_point = [bounds[0]//2, bounds[1]//2]
        # assumption is made that the action is valid
        action_loc = self.belief_map.get_action_loc(action, curr_bot_loc)

        displacement = [(mid_point[0] - action_loc[0]), (mid_point[1] - action_loc[1])]

        partial_info = self.partial_info_matrices[-1]

        for x in range(len(partial_info)):
            if (x + displacement[0]) < bounds[0]:
                for y in range(len(partial_info)):
                    if (y + displacement[1] < bounds[1]):
                        action_matrix[(x + displacement[0]), (y + displacement[1])] = partial_info[x, y]

        """"
        [[2 2 2 2 2]
        [2 0 0 0 2]
        [2 0 0 0 2]
        [2 0 0 0 2]
        [2 2 2 2 2]]

        action = forward

        [[1 2 2 2 2]
        [1 2 0 0 0]
        [1 2 0 0 0]
        [1 2 0 0 0]
        [1 2 2 2 2]]

        Remember that the x axis here is the left corner going downwards.
        Y axis is going to the right.
        So an action of forward where (y-1) means that the action will be to the left of robot mid-point from my frame.
        """
        if get_matrix:
            return action_matrix
            
        # refractor: this should not be appending to a final list. 
        # this should just return the action matrix and the appending should happen elsewhere
        self.action_matrices.append(action_matrix)

            
    def append_action_matrix(self, matrix):
        self.action_matrices.append(matrix)

    def get_partial_info_matrices(self):
        return self.partial_info_matrices

    def get_action_matrices(self):
        return self.action_matrices

    def get_path_matrices(self):
        return self.path_matrices
    
    def get_comm_path_matrices(self):
        return self.comm_path_matrices



