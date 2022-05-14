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



