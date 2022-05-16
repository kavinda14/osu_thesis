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
def generate_valid_neighbors(curr_state, state_sequence, bot_belief_map):
    neighbors = list()
    curr_bot_loc = curr_state.get_loc()

    sequence = [state.get_loc() for state in state_sequence]
    actions = ['left', 'right', 'forward', 'backward']
    for action in actions:
        # new_loc is the loc the bot will end up after taking action
        valid, new_loc = bot_belief_map.is_valid_action(action, curr_bot_loc, mcts=True)
        if valid and new_loc not in sequence:
            neighbors.append(State(action, new_loc))

    # condition added because rollout_random ends up in spot with no neighbors sometimes
    if len(neighbors) == 0:
        while True:
            action_idx = randint(0, len(actions)-1)
            action = actions[action_idx]
            new_loc = bot_belief_map.get_action_loc(action, curr_bot_loc)
            if bot_belief_map.is_valid_loc(new_loc[0], new_loc[1]):
                neighbors.append(State(action, new_loc))
                break

    return neighbors
