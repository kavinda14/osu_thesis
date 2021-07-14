class Robot:
    def __init__(self, x, y, bounds, map):
        # Variables that changes
        self.x_loc = x
        self.y_loc = y
        self.path = list() # Variable to track where the robot has been
        self.index = 0

        # Static variables
        self.start_loc = self.x_loc, self.y_loc
        self.velocity = 1.0
        self.sensing_range = 1.0
        self.lim = bounds
        self.map = map

    def reset_robot(self):
        self.x_loc = self.start_loc[0]
        self.y_loc = self.start_loc[1]
        self.path = list()
        self.index = 0
    
    def check_valid_loc(self):
        x = self.x_loc
        y = self.y_loc
        in_bounds = (x >= self.lim[0] and x <= self.lim[1] and y >= self.lim[0] and y <= self.lim[1])

        # Check if new location is intersecting with obstacles from map
        for loc in self.map.unobs_occupied:
            if x == loc[0] and y == loc[1]:
                print("Invalid Location {}".format(loc))
                return False

        return in_bounds

    def check_new_loc(self, x_loc, y_loc):
        x = x_loc
        y = y_loc
        #in_bounds = (x >= self.lim[0] and x <= self.lim[1] and y >= self.lim[0] and y <= self.lim[1])
        in_bounds = (x >= 0 and x <= self.lim[0] and y >= 0 and y <= self.lim[1])


        # Check unobs_occupied from map
        for loc in self.map.unobs_occupied:
            if x == loc[0] and y == loc[1]:
                return False

        return in_bounds

    def get_loc(self):
        return (self.x_loc, self.y_loc)

    def set_loc(self, x_loc, y_loc):
        self.x_loc = x_loc
        self.y_loc = y_loc

    def check_valid_move(self, direction, updateState=False):
        """ Checks if the direction is valid
        direction (str): "left", "right", "up", "down" directions to move the robot
        updateState (bool): if True, function also moves the robot if direction is valid
                            otherwise, only perform validity check without moving robot
        """
        # Just don't move
        if not direction:
            return True

        if direction == 'left':
            valid = self.check_new_loc(self.x_loc-1, self.y_loc)
            if valid and updateState:
                self.x_loc -= 1

        elif direction == 'right':
            valid = self.check_new_loc(self.x_loc+1, self.y_loc)
            if valid and updateState:
                self.x_loc += 1

        elif direction == 'backward':
            valid = self.check_new_loc(self.x_loc, self.y_loc+1)
            if valid and updateState:
                self.y_loc += 1

        elif direction == 'forward':
            valid = self.check_new_loc(self.x_loc, self.y_loc-1)
            if valid and updateState:
                self.y_loc -= 1
        else:
            raise ValueError(f"Robot received invalid direction: {direction}!")

        return valid

    def move(self, direction):
        """ Move the robot while respecting bounds"""
        self.check_valid_move(direction, updateState=True)

    ###Functions used when evaluated a final path
    def set_path(self, path):
        self.path = path

    def follow_direction_path(self):
        """ Select direction that move robot along a pre-computed path"""
        direction = None
        if self.index+1 >= len(self.final_path):
            return direction
        direction = self.final_path[self.index]
        self.index += 1
        return direction

    def follow_path(self):
        """ Select direction that move robot along a pre-computed path"""
        direction = None
        if self.index+1 >= len(self.final_path):
            return direction

        current_loc = self.final_path[self.index].location
        next_loc = self.final_path[self.index+1].location
        if next_loc[0] == current_loc[0]-1:
            direction = "left"
        if next_loc[0] == current_loc[0]+1:
            direction = "right"
        if next_loc[1] == current_loc[1]-1:
            direction = "forward"
        if next_loc[1] == current_loc[1]+1:
            direction = "backward"

        self.index += 1
        return direction