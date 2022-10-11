# What is the multi-robot_softmax branch for?
This branch contains the **fourth multi-robot** implementation.

This code was modified to include the new planner "Action-CNN" that outputs actions instead of the reward by using the CNN.

Run `main.py` with the following command for **evaluation**:
```bash
python3 main.py eval *name_of_file*
```

Run `main.py` with the following command to **generate training data**:
```bash
python3 main.py gen_data *name_of_file*
```
