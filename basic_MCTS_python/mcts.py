'''
Basic MCTS implementation
Graeme Best
Oregon State University
Jan 2020
'''

from tree_node import TreeNode
import reward
from cost import cost
from rollout import rollout_network, rollout_greedy, rollout_random
import copy
import random
import math
import pickle
from util import State, generate_valid_neighbors


# def mcts(action_set, budget, max_iterations, exploration_exploitation_parameter, robot, input_map):
def mcts(budget, max_iterations, exploration_exploitation_parameter, robot, sensor_model, world_map, rollout_type, reward_type, neural_model, 
         debug_mcts_reward_greedy_list, debug_mcts_reward_network_list, CONF, json_comp_conf, device=False):

    ################################
    # Setup
    # *what is a sequence?
    # -> a list to hold the actions, which can be used to calculate the budget left
    # in our case, this sequence is passed to every other node to keep track of history
    # the history will allow us to check if a particular state was previously visited
    # world_map = copy.deepcopy(input_map)
    start_sequence = [State('root', robot.get_loc())]
    # what is action_set?
    # -> there is an action object created in main.py
    unpicked_child_actions = generate_valid_neighbors(start_sequence[0], start_sequence, robot)
    # unpicked_child_actions = copy.deepcopy(action_set)
    # root is created when mcts is run
    root = TreeNode(parent=None, sequence=start_sequence, budget=budget, unpicked_child_actions=unpicked_child_actions, coords=robot.get_loc())
    list_of_all_nodes = []
    list_of_all_nodes.append(root) # for debugging only

    ################################
    # Main loop
    # this determines the depth of the tree
    for iter in range(max_iterations):

        # print('Iteration: ', iter)

        ################################
        # Selection and Expansion
        # move recursively down the tree from root
        # then add a new leaf node
        current = root
        while True: 

            # Are there any children to be added here?
            #### SELECTION
            if current.unpicked_child_actions: # if not empty
                
                # does this mean that selection happens randomly?

                # Pick one of the UNPICKED children that haven't been added
                # Remember that these child actions are "State" objects - contains action + location
                # Do this at random
                num_unpicked_child_actions = len(current.unpicked_child_actions)
                if num_unpicked_child_actions == 1:
                    child_index = 0
                else:
                    child_index = random.randint(0,num_unpicked_child_actions-1)
                child_action = current.unpicked_child_actions[child_index] # even though it says action, it is a State() object that contains action as an attribute
                child_loc = child_action.get_location()

                # Remove the child form the unpicked list
                del current.unpicked_child_actions[child_index]

                # Setup the new action sequence
                # what does it mean to add something to a treenode sequence?
                # -> is it the order of traversal down the tree?
                new_sequence = copy.deepcopy(current.sequence)
                new_sequence.append(child_action)
                new_budget_left = budget - cost(new_sequence)

                # Setup the new child's unpicked children
                # Remove any over budget children from this set
                new_unpicked_child_actions = generate_valid_neighbors(child_action, new_sequence, robot)
                def is_overbudget(a):
                    seq_copy = copy.deepcopy(current.sequence)
                    seq_copy.append(a)
                    return cost(seq_copy) > budget
                
                # for the next node created, this adds actions to it only if there is budget left to do those actions
                new_unpicked_child_actions = [a for a in new_unpicked_child_actions if not is_overbudget(a)]

                # Create the new node and add it to the tree
                #### EXPANSION
                new_child_node = TreeNode(parent=current, sequence=new_sequence, budget=new_budget_left, unpicked_child_actions=new_unpicked_child_actions, coords=child_loc)
                current.children.append(new_child_node)
                current = new_child_node
                list_of_all_nodes.append(new_child_node) # for debugging only

                break # don't go deeper in the tree...

            else:
                
                # All possible children already exist
                # Therefore recurse down the tree
                # using the UCT selection policy

                if not current.children:

                    # Reached planning horizon -- just do this again
                    break
                else:
                    # this is the point where the selection recurses down the tree till leaf node is reached

                    # Define the UCB
                    def ucb(average, n_parent, n_child):
                        return average + exploration_exploitation_parameter * math.sqrt( (2*math.log(n_parent)) / float(n_child) )

                    # Pick the child that maximises the UCB
                    n_parent = current.num_updates
                    best_child = -1
                    best_ucb_score = 0
                    for child_idx in range(len(current.children)):
                        child = current.children[child_idx]
                        ucb_score = ucb(child.average_evaluation_score, n_parent, child.num_updates)
                        if best_child == -1 or (ucb_score > best_ucb_score):
                            best_child = child
                            best_ucb_score = ucb_score

                    # Recurse down the tree
                    current = best_child

        ################################
        # Rollout
        if rollout_type == 'random':
            rollout_sequence = rollout_random(subsequence=current.sequence, budget=budget, robot=robot)
        elif rollout_type == 'greedy':
            rollout_sequence = rollout_greedy(subsequence=current.sequence, budget=budget, robot=robot, sensor_model=sensor_model, world_map=world_map)
        else: # all networks will run this
            rollout_sequence = rollout_network(subsequence=current.sequence, budget=budget, robot=robot, sensor_model=sensor_model, world_map=world_map, neural_model=neural_model, device=device)

        # TEST TO CHECK IF GREEDY AND NETWORK REWARDS ARE LINEAR
        debug_reward_greedy = reward.reward_greedy(rollout_sequence, sensor_model, world_map, oracle=True)
        debug_reward_network = reward.reward_network(rollout_sequence, sensor_model, world_map, neural_model, device=device)
        debug_mcts_reward_greedy_list.append(debug_reward_greedy)
        debug_mcts_reward_network_list.append(debug_reward_network)

        # pickle progress
        filename1 = CONF[json_comp_conf]["pickle_path"] + "debug_reward_greedy_list"
        filename2 = CONF[json_comp_conf]["pickle_path"] + "debug_reward_network_list"
        outfile = open(filename1,'wb')
        pickle.dump(debug_mcts_reward_greedy_list, outfile)
        outfile.close()
        outfile = open(filename2,'wb')
        pickle.dump(debug_mcts_reward_network_list, outfile)
        outfile.close()

        if reward_type == 'random':
            rollout_reward = reward.reward_random(rollout_sequence)
        elif reward_type == 'greedy':
            rollout_reward = reward.reward_greedy(rollout_sequence, sensor_model, world_map)
        else: # all networks will run this
            rollout_reward = reward.reward_network(rollout_sequence, sensor_model, world_map, neural_model, device=device)

        ################################
        #### BACK PROPAGATION
        # update stats of all nodes from current back to root node
        parent = current
        while parent: # is not None

            # Update the average
            # parent.updateAverage(rollout_reward)
            parent.updateAverage(rollout_reward)

            # Recurse up the tree
            parent = parent.parent

    ################################
    # Extract solution
    # calculate best solution so far
    # by recursively choosing child with highest average reward
    current = root
    # print('root :', current.get_children())
    # for node in current.get_children():
    #     print('child loc: ', node.get_coords())

    # while current.children: # is not empty

    #     # Find the child with best score
    #     best_score = 0
    #     best_child = -1
    #     for child_idx in range(len(current.children)):
    #         child = current.children[child_idx]
    #         score = child.average_evaluation_score
    #         if best_child == -1 or (score > best_score):
    #             best_child = child
    #             best_score = score

    #     current = best_child

    best_score = 0
    best_child = -1

    solution_locs = list()
    for child in current.children: # is not empty
        solution_locs.append(child.get_coords())

        # Find the child with best score
        score = child.average_evaluation_score
        # print('score: ', [child.get_coords(), score])
        if best_child == -1 or (score > best_score):
            best_child = child
            best_score = score

    solution = current.sequence
    winner_node = best_child
    winner_loc = winner_node.get_coords()

    return [solution, solution_locs, root, list_of_all_nodes, winner_node, winner_loc]
