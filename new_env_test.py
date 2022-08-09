from GroundTruthMap import GroundTruthMap
from BeliefMap import BeliefMap

if __name__ == "__main__":

    # BOUNDS = [41, 41]
    BOUNDS = [21, 21]
    OCC_DENSITY = 6

    for i in range(10):
        ground_truth_map = GroundTruthMap(BOUNDS, OCC_DENSITY)
        belief_map = BeliefMap(BOUNDS)
        ground_truth_map.visualize(1)