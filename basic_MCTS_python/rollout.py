'''
Basic MCTS implementation
Graeme Best
Oregon State University
Jan 2020
'''

from cost import cost
import random
import copy
# from mcts import State

# def rollout(subsequence, action_set, budget):
#     # Random rollout policy
#     # Pick random actions until budget is exhausted
#     num_actions = len(action_set)
#     if num_actions <= 0:
#         raise ValueError('rollout: num_actions is ' + str(num_actions))
#     sequence = copy.deepcopy(subsequence)
#     while cost(sequence) < budget:
#         r = random.randint(0,num_actions-1)
#         sequence.append(action_set[r])

#     return sequence

class State():
    def __init__(self, action, location):
        self.action = action
        self.location = location

    def get_action(self):
        return self.action
    
    def get_location(self):
        return self.location

def rollout(subsequence, budget):
    # Random rollout policy
    # Pick random actions until budget is exhausted
    action_set = list()
    actions = ['left', 'right', 'forward', 'backward']
    for action in actions:
        action_set.append(State(action, [random.randint(0, 20), random.randint(0, 20)]))
        
    if len(action_set) <= 0:
        raise ValueError('rollout: num_actions is ' + str(num_actions))
    sequence = copy.deepcopy(subsequence)
    while cost(sequence) < budget:
        r = random.randint(0, len(action_set)-1)
        sequence.append(action_set[r])

    return sequence
