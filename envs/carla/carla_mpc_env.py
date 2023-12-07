

import os, sys
import time
import random
import numpy as np 
import gym
import copy

# carla
import carla
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CUR_DIR)
from carla_env import CarlaEnv
from mpc import MPCAgent


class CarlaMPCEnv(CarlaEnv):
    """
        - State: 
        - Action: [steer, brake, throttle] 3-dim float 
        - Reward:  
        - Info:
    """

    def __init__(self, is_render=False, random_seed=2022, task_name="straight", perturb_spec={}, port=2000, domain_random=False): 

        print("\nDomain Random: ", domain_random)
        super().__init__(is_render, random_seed, task_name, perturb_spec, port, domain_random)

        self.mpc_controller = MPCAgent()
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32) 

        # M: action of rl policy on model parameters
        ACT_DIM = 64
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(ACT_DIM,), dtype=np.float32) # length

        # M: rescale different actions
        self.action_scale = np.array([1.0]*ACT_DIM) * 1.0

    def reset(self, start_id=None): 
        self.state, info = super().reset(start_id)
        self.info = info

        # obs = [s_t, a_{t-1}]
        state = copy.deepcopy(self.state)
        state = np.concatenate([state, np.zeros((3))])
        return state

    def step(self, action, action_info={}): 

        # 0. rescale action
        action = action * self.action_scale

        # 1. setup MPC transitions
        self.mpc_controller.set_parameters(action)

        # 2. execute mpc agent
        control, trj, cost = self.mpc_controller.act(self.state, self.info)

        # 3. interact with environment
        if self.is_render:
            next_state, reward, done, info = super().step(control, {"x_trj": trj})
        else:
            next_state, reward, done, info = super().step(control)
        self.state = next_state
        self.info = info

        # obs = [s_t, a_{t-1}]
        state = copy.deepcopy(self.state)
        state = np.concatenate([state, info["control"]])

        # M: only return scalars?
        return_info = {'vel_error':info['vel_error'], 'route_error':info['route_error'], 'goal_error':info['goal_error'], 'control_action':info['control']}
        return state, reward, done, return_info


# ==============================================================================
# -- Testing ---------------------------------------------------------
# ==============================================================================

if __name__ == "__main__":

    # Initialize Carla MPC Environment
    env = CarlaEnv(is_render=False, random_seed=2022, task_name="straight")

    for ep in range(10):
        print(f"=== Episode: {ep} ===")
        state = env.reset()
        times = []
        for i in range(100):
           action = np.random.rand(1)
           t = time.time()
           next_state, reward, done, info = env.step(action)
           times.append(time.time()-t)

           # print(f"=== Step {i} ===")
           # print(f"state: {state} action: {action} reward: {reward} next_state: {next_state}") 
           # images = env.render()
           # print(len(images))
           state = next_state

        print(f"AVG_TIME/STEP: {np.mean(times)} FPS: {1.0/np.mean(times)} MAX_TIME: {np.max(times)}")

    env.close()



