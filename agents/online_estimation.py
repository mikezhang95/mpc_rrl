import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from agents import Agent
from agents.system_id import VehicleModel
import utils

import hydra


class OnlineEstimationAgent(Agent):
    """DUMMY algorithm."""
    def __init__(self, obs_dim, action_dim, device,
                 model_lr, model_update_epochs, replay_buffer_size,
                 action_range, update_frequency,
                 batch_size):
        super().__init__()

        self.action_dim = action_dim
        self.device = device

        # M: customized system identification model
        self.init_system_model = VehicleModel(self.device)
        self.system_model = VehicleModel(self.device)
        self.system_model.w3.requires_grad = True
        self.batch_size = batch_size
        self.num_epochs = model_update_epochs
        self.model_lr = model_lr
        self.optimizer = torch.optim.Adam([self.system_model.w3], lr=self.model_lr)  # only w3 is trainable 64-dim

    def update_parameters(self, transitions):
        print("\nStarting to do online estimations...")
        # batchify transitions
        len_transitions = len(transitions["obs"])
        num_batch = len_transitions // self.batch_size
        for epoch in range(self.num_epochs):
            loss = 0.0
            index = np.arange(len_transitions)
            np.random.shuffle(index)
            for i in range(num_batch):
                batch_obs = torch.Tensor(np.array(transitions["obs"])[index[i*self.batch_size:(i+1)*self.batch_size]]).to(self.device)
                batch_control = torch.Tensor(np.array(transitions["control"])[index[i*self.batch_size:(i+1)*self.batch_size]]).to(self.device)
                batch_next_obs = torch.Tensor(np.array(transitions["next_obs"])[index[i*self.batch_size:(i+1)*self.batch_size]]).to(self.device)
                
                pred_next_obs = self.system_model(batch_obs, batch_control)
                
                self.optimizer.zero_grad()
                loss_fn = nn.MSELoss()
                si_loss = loss_fn(pred_next_obs, batch_next_obs[:, [2,4]])
                si_loss.backward()
                self.optimizer.step()
                loss += si_loss.item()
            print(f"epoch: {epoch} supervised learning loss: {loss}")
        if num_batch > 0:
            return loss / num_batch
        else:
            return loss

    def calc_act(self, ):
        with torch.no_grad():
            a = self.system_model.w3.flatten() / self.init_system_model.w3.flatten() - 1 
        return a.cpu().numpy().reshape((self.action_dim))

    def act(self, obs, input_info={}, sample=False, action=None):
        return self.calc_act(), {}

    def predict(self, obs, **other_params):
        return self.calc_act(), {}

