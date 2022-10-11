# What is the multi-robot_scalability branch for?
This branch contains the **third multi-robot** implementation.

This code was modified to test different increments of the robot team size.

The robot team size can be modified by changing the constant `NUM_ROBOTS`.

The previous implmentation of this can be found in the branch `multi-robot_optimized`.

Run `main.py` with the following command for **evaluation**:
```bash
python3 main.py eval *name_of_file*
```

Run `main.py` with the following command to **generate training data**:
```bash
python3 main.py gen_data *name_of_file*
```
