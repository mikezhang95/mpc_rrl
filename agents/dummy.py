import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from agents import Agent
import utils

import hydra


class DummyAgent(Agent):
    """DUMMY algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature, robust_method="none", robust_coef=1e-3):
        super().__init__()
        self.action_dim = action_dim
        self.training = False

    def act(self, obs, input_info={}, sample=False, action=None):
        return np.zeros((self.action_dim), dtype=np.float32), {}

    def predict(self, obs, **other_params):
        return np.zeros((self.action_dim), dtype=np.float32), {}
