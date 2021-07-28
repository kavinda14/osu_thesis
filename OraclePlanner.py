import random

def random_planner(robot):
    actions = ['left', 'right', 'backward', 'forward']
    valid_move = False
    action = ''

    while not valid_move:
        action = random.choice(actions)
        valid_move = robot.check_valid_move(action) 
    
    return action
