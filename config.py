import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

SCENARIOS = {
    "normal": (0, 0),
    "type-a": (4, 0),
    "type-b": (0, 4),
    "mixed": (2, 2),
    "stress": (3, 3),
}

# 0. simulation settings
MAX_STEPS = 2000
USE_FIXED_SEED = True 
RANDOM_SEED = 70
MAX_LOG_FILES = 30           # Maximum number of log files to keep
SAVE_TRAJECTORY_PNG = True
SAVE_ANIMATION_MP4 = True    # Enable MP4 video generation
DATA_LOG_DIR = str(PROJECT_ROOT / "logs" / "data")
ANALYSIS_LOG_DIR = str(PROJECT_ROOT / "logs" / "analysis")
VISUAL_LOG_DIR = str(PROJECT_ROOT / "logs" / "visuals")
VIDEO_LOG_DIR = str(PROJECT_ROOT / "logs" / "videos")
CONTROL_LOG_DIR = str(PROJECT_ROOT / "logs" / "control")
REPAIR_ENABLED = True        # Enable alarm-driven shepherd repair intervention
GOAL_RETENTION_ENABLED = True   # Soft containment for sheep near the goal edge
GOAL_RETENTION_INNER_FRACTION = 0.75
GOAL_RETENTION_WEIGHT = 0.5
GOAL_RETENTION_VELOCITY_DAMPING = 0.90

# High-level integration settings. "rule" remains the default controller.
CONTROLLER_KIND = "rule"
MOCK_LLM_LATENCY_FRAMES = 30
HIGH_LEVEL_HOLD_FRAMES = 45
HIGH_LEVEL_MIN_HOLD_FRAMES = 15
HIGH_LEVEL_MAX_HOLD_FRAMES = 60
HIGH_LEVEL_MIN_DWELL_FRAMES = 15
HIGH_LEVEL_MAX_RESPONSE_AGE_FRAMES = 600
HIGH_LEVEL_RETRY_FRAMES = 30
HIGH_LEVEL_CONTEXT_FRAMES = 30
HIGH_LEVEL_FORECAST_FRAMES = 30

# Repair-target selection. Both repair modes keep a target briefly and switch
# only when another sheep is meaningfully more urgent.
REPAIR_TARGET_SELECTION_ENABLED = True
COHESION_TARGET_MIN_HOLD_FRAMES = 30
HARD_TO_GUIDE_TARGET_MIN_HOLD_FRAMES = 30
REPAIR_TARGET_SWITCH_MARGIN = 3.0
REPAIR_TARGET_LOOKAHEAD_FRAMES = 20
REPAIR_TARGET_OUTSIDE_GOAL_FIRST = True

# Shared settings for Responses API providers.
LLM_API_TIMEOUT_SECONDS = 20.0
LLM_MAX_OUTPUT_TOKENS = 200
LLM_PENDING_POLL_SECONDS = 0.05
LLM_CACHE_ENABLED = True

# Optional API providers. Environment variables can override these defaults.
OPENAI_MODEL = "gpt-5-mini"
OPENAI_BASE_URL = None
OPENAI_CACHE_PATH = str(PROJECT_ROOT / "logs" / "llm_cache" / "openai_responses.jsonl")
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CACHE_PATH = str(PROJECT_ROOT / "logs" / "llm_cache" / "deepseek_responses.jsonl")
DEEPSEEK_REASONING_EFFORT = "none"  # This short mode-selection task needs no CoT

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

# 5. Boids model constants
SEPARATION_RADIUS_NORMAL = 3.0 # normal sheep's separation sensing radius
GOAL_RADIUS = 20.0             # radius to consider the goal as reached
DOG_DRIVE_DISTANCE = 25.0      # distance behind the flock the dog aims for
HARD_TO_GUIDE_DISTANCE = 3.0   # close standoff behind one hard-to-guide sheep
COHESION_REPAIR_DISTANCE = 10.0 # standoff outside the most dispersed sheep
COHESION_THRESHOLD = 25.0      # threshold of max distance to COM to trigger collect mode
BOUNDARY_MARGIN = 5.0          # distance from the wall to start repelling
DOG_SENSING_RANGE = 50.0       # radius within which sheep react to the dog
EPSILON = 1e-6                 # small number to avoid division by zero

# 6. abnormal sheep settings
NUM_ABNORMAL_A = 3             # number of abnormal sheep A (Unresponsive)
NUM_ABNORMAL_B = 3             # number of abnormal sheep B (Dispersing)
WEIGHT_DOG_REPULSION_A_FACTOR = 0.15 # type A dog fear factor (vs normal)
INERTIA_FACTOR_A = 0.95        # type A inertia retention (higher is more sluggish)
WEIGHT_COHESION_B_FACTOR = 0.15  # type B cohesion force factor (vs normal)
WEIGHT_SEPARATION_B_FACTOR = 1.5 # type B separation force factor (vs normal)
SEPARATION_RADIUS_B = 6.0      # type B's separation sensing radius (larger)

# 7. danger detection settings
DANGER_DISPERSION_THRESHOLD = 40.0 # dispersion radius > this value triggers an alarm
DANGER_MEAN_SPREAD_THRESHOLD = 15.0 # average flock spread > this value triggers an alarm
DANGER_STAGNATION_FRAMES = 150     # number of consecutive frames to check for stagnation
DANGER_STAGNATION_THRESHOLD = 1.0  # if progress is less than this over the frames, it's stagnation
