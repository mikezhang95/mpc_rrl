


import os, sys
import numpy as np
import pickle
import time
import copy

import torch
import torch.nn.functional as F

MPC_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../envs/carla")
sys.path.append(MPC_DIR)
from constants import *


class VehicleModel(torch.nn.Module):

    def __init__(self, device): 

        super().__init__()
        self.device = device

        MODEL_NAME = "bicycle_model_100ms_20000_v4_jax"
        model_path=f"{MPC_DIR}/net_{MODEL_NAME}.model"
        NN_W1, NN_W2, NN_W3, NN_LR_MEAN = pickle.load(open(model_path, mode="rb"))

        self.w1 = torch.FloatTensor(NN_W1).to(self.device) # [32,5]
        self.w2 = torch.FloatTensor(NN_W2).to(self.device) # [32,32]
        self.w3 = torch.FloatTensor(NN_W3).to(self.device) # [2,32]
        self.lr_mean = torch.FloatTensor([NN_LR_MEAN]).to(self.device) 


    def nn3(self, x, action):
        
        x = torch.matmul(x, self.w1.transpose(1,0))
        x = torch.tanh(x)

        x = torch.matmul(x, self.w2.transpose(1,0))
        x = torch.tanh(x)

        # w3 = self.w3 * ( 1 + 2.0 * action.reshape(-1, 2, 1) ) # if action: Bx2
        w3 = self.w3 * ( 1 + 1.0*action.reshape(-1, self.w3.shape[0], self.w3.shape[1]) ) # if action: Bx64

        x = torch.matmul(x.unsqueeze(1), w3.transpose(1,2)).squeeze(1)
        return x


    def forward(self, obs, control, action=None):

        if action is None:
            action = torch.zeros((obs.shape[0], 64), dtype=torch.float, device=obs.device)

        v = obs[:, 2:3]
        v_sqrt = torch.sqrt(v)
        beta = obs[:, 4:5]
        steering, throttle, brake = control[:, 0:1], control[:, 1:2], control[:, 2:3]

        x1 = torch.cat([v_sqrt, 
                    torch.cos(beta),
                    torch.sin(beta),
                    steering, 
                    throttle,
                    brake], dim=-1)

        x2 = torch.cat([v_sqrt, 
                    torch.cos(beta),
                    -torch.sin(beta),
                    -steering, 
                    throttle,
                    brake], dim=-1)

        x1 = self.nn3(x1, action)
        x2 = self.nn3(x2, action)

        deriv_v = ( x1[:, 0:1]*(2*v_sqrt+x1[:, 0:1]) + x2[:, 0:1]*(2*v_sqrt+x2[:, 0:1]) ) / 2 / DT # x1[0]+x2[0]
        deriv_beta = ( x1[:, 1:2] - x2[:, 1:2] ) / 2 / DT

        pred_v =  v + deriv_v  * DT
        pred_beta = beta + deriv_beta  * DT
        pred_obs = torch.cat([
                            pred_v,
                            pred_beta
                            ], dim=-1)
        return pred_obs


