from Map import Map
from Robot import Robot

if __name__ == "__main__":

    bounds = [10, 10]
    map = Map(bounds, 3)
    print("unobs_occupied: ", map.unobs_occupied)
    print("unobs_free: ", map.unobs_free)

    robot = Robot(0, 0, bounds, map)

    robot.move('right')

    print("Robot location: ", robot.get_loc())
    