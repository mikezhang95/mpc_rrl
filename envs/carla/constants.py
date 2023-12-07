

import numpy as np

# ==============================================================================
# -- Define constants ---------------------------------------------------------
# ==============================================================================

# Carla
MAX_STEPS = 100
DT_ = 0.1 # simulation time per step [s] ! keep it fixed for pretrained model
N_DT = 1 # steps per mpc control 
FPS = 1.0 / DT_
DT = DT_ * N_DT # simualtion time per mpc control
WAYPOINT_BUFFER_LEN = 100 
WAYPOINT_INTERVAL = 1.0 # [m]
FUTURE_WAYPOINTS_AS_STATE = 30
DISTANCE_TO_SUCCESS = 1.0 # [m]

RES_X = 1920
RES_Y = 1080

# MPC
N_X = 5 # state dim
N_U = 3 # act dim
MPC_HORIZON = 20
TARGET_SPEED = 8.0 

# TODO: cost normalization
TIME_STEPS_RATIO = MPC_HORIZON / FUTURE_WAYPOINTS_AS_STATE 
TARGET_RATIO = FUTURE_WAYPOINTS_AS_STATE * WAYPOINT_INTERVAL / (6*np.pi) 

# ==============================================================================
