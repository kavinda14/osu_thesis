from random import randint

class State():
    def __init__(self, action, loc):
        self.action = action
        self.loc = loc

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

    def get_loc(self):
        return self.loc


# returns valid State objects (contains action and location) from a given position
def generate_valid_neighbors(curr_state, bot_belief_map):
    neighbors = list()
    curr_bot_loc = curr_state.get_loc()

    actions = ['left', 'right', 'forward', 'backward']
    for action in actions:
        # new_loc is the loc the bot will end up after taking action
        valid, new_loc = bot_belief_map.is_valid_action(action, curr_bot_loc, mcts=True)
        if valid:
            neighbors.append(State(action, new_loc))

    return neighbors
