import numpy as np

# 0. simulation settings
MAX_STEPS = 2000
USE_FIXED_SEED = False  
MAX_LOG_FILES = 30           # Maximum number of log files to keep
SAVE_TRAJECTORY_PNG = True
SAVE_ANIMATION_GIF = False
DATA_LOG_DIR = "logs/data"
VISUAL_LOG_DIR = "logs/visuals"

# 1. space and goal settings
SPACE_SIZE = 200.0
GOAL_MARGIN = 20.0

# 2. sheep settings
NUM_SHEEP = 20
SHEEP_MAX_SPEED = 0.7
SHEEP_START_MARGIN = 30.0
SHEEP_START_RADIUS = 15.0

# 3. shepherd settings
NUM_DOGS = 1
DOG_MAX_SPEED = 1.5  # the dog must run faster than the sheep
DOG_START_MARGIN = 10.0

# 4. physical rule weights (control the strength of the behavior)
WEIGHT_COHESION = 0.01       # the sheep's desire to converge to the center
WEIGHT_DOG_REPULSION = 10.0  # the sheep's fear of the dog
WEIGHT_SEPARATION = 2.0      # prevent sheep from overlapping
WEIGHT_BOUNDARY = 1.0        # the boundary's repulsion force

# 5. model constants
GOAL_RADIUS = 10.0           # radius to consider the goal as reached
DOG_DRIVE_DISTANCE = 25.0    # distance behind the flock the dog aims for
COHESION_THRESHOLD = 25.0    # threshold of max distance to COM to trigger collect mode
BOUNDARY_MARGIN = 5.0        # distance from the wall to start repelling
DOG_SENSING_RANGE = 50.0     # radius within which sheep react to the dog
EPSILON = 1e-6               # small number to avoid division by zero

# 6. abnormal sheep settings
NUM_ABNORMAL_A = 2           # number of abnormal sheep A (ignores dog)
NUM_ABNORMAL_B = 2           # number of abnormal sheep B (low cohesion)
WEIGHT_DOG_REPULSION_A = 0.0 # type A completely ignores dog repulsion
WEIGHT_COHESION_B = 0.01     # type B has very low cohesion