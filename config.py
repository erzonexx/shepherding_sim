import numpy as np

# 0. simulation settings
MAX_STEPS = 2000
# 1. space and goal settings
SPACE_SIZE = 200.0
GOAL_POS = np.random.uniform(20.0, SPACE_SIZE - 20.0, 2)

# 2. sheep settings
NUM_SHEEP = 20
SHEEP_MAX_SPEED = 1.0
SHEEP_START_CENTER = np.random.uniform(30.0, SPACE_SIZE - 30.0, 2)
SHEEP_START_RADIUS = 15.0

# 3. shepherd settings
NUM_DOGS = 1
DOG_MAX_SPEED = 1.3  # the dog must run faster than the sheep
DOG_START_POS = np.random.uniform(10.0, SPACE_SIZE - 10.0, 2)

# 4. physical rule weights (control the strength of the behavior)
WEIGHT_COHESION = 0.02       # the sheep's desire to converge to the center
WEIGHT_DOG_REPULSION = 5.0   # the sheep's fear of the dog
WEIGHT_BOUNDARY = 1.0        # the boundary's repulsion force

# 5. model constants
GOAL_RADIUS = 10.0           # radius to consider the goal as reached
DOG_DRIVE_DISTANCE = 25.0    # distance behind the flock the dog aims for
BOUNDARY_MARGIN = 5.0        # distance from the wall to start repelling
EPSILON = 1e-6               # small number to avoid division by zero