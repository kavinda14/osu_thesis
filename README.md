# What is the multi-robot_optimized branch for?
This branch contains the **second multi-robot** implementation.

This code was refractored to:
1) make it more readable and maintainable 
2) separate the belief and ground truth map classes

The previous implmentation of this can be found in the branch `multi-robot`.

Run `main.py` with the following command for **evaluation**:
```bash
python3 main.py eval *name_of_file*
```

Run `main.py` with the following command to **generate training data**:
```bash
python3 main.py gen_data *name_of_file*
```
