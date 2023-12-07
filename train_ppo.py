#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import importlib
from omegaconf import OmegaConf

import hydra
import stable_baselines3
import sb3_contrib
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from logger import Logger
import utils

def make_env(cfg, rank=0):
    def _init():
        cfg.env.params.port = cfg.env.params.port + 2*rank
        env = utils.make_env(cfg)
        env = Monitor(env)
        return env
    return _init

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        print(f'config: {self.cfg}')

        self.logger = Logger(self.work_dir,
                             save_tb=False, # cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # vectorize environment
        num_processes = cfg.num_train_env
        # self.train_env = DummyVecEnv([make_env(cfg, i) for i in range(num_processes)])
        self.train_env = SubprocVecEnv([make_env(cfg, i) for i in range(num_processes)])

        self.eval_env = utils.make_env(cfg)
        cfg.agent.params.tensorboard_log = self.work_dir

        # - M: cannot initialize with hydra due to env
        # -    so that initialize handly
        # cfg.agent.params.env = self.train_env
        # self.agent = hydra.utils.instantiate(cfg.agent)
        cfg.agent.params.env = "tmp_env"
        agent = OmegaConf.to_container(cfg.agent, resolve=True)
        # PPO params
        agent_params = agent['params'] 
        agent_params['env'] = self.train_env

        # PPO class
        module_class = agent['class'].split('.')
        module_name = '.'.join(module_class[:-1])
        class_name = module_class[-1]
        agent_module = importlib.import_module(module_name)
        agent_class = getattr(agent_module, class_name)

        # initialize agent 
        self.agent = agent_class(**agent_params)

        self.step = 0

    # def evaluate(self):
    #     average_episode_reward = 0
    #     episode_starts = np.ones((num_envs,), dtype=bool)
    #     for episode in range(self.cfg.num_eval_episodes):
    #         obs = self.eval_env.reset()
    #         if type(obs) == tuple: # special for carla environment
    #             obs, info = obs
    #         last_hid = self.agent.reset()
    #         done = False
    #         episode_reward = 0
    #         # cell and hidden state of the RNN
    #         rnn_state = None
    #         # Episode start signals are used to reset the lstm states
    #         episode_start = np.ones((1,), dtype=bool)
    #         while not done:
    #             action, rnn_state = model.predict(obs, state=rnn_state, episode_start=episode_start, deterministic=True)
    #             obs, reward, done, _ = self.eval_env.step(action)
    #             episode_reward += reward
    #             episode_start = done
    #         average_episode_reward += episode_reward

    #     average_episode_reward /= self.cfg.num_eval_episodes
    #     self.logger.log('eval/episode_reward', average_episode_reward, self.step)
    #     self.logger.dump(self.step)
    #     return average_episode_reward


    def run(self):

        checkpoint_callback = CheckpointCallback(
                save_freq=self.cfg.eval_frequency * 5, 
                save_path=os.path.join(self.work_dir, "checkpoints"), 
                save_replay_buffer=False,
                save_vecnormalize=False
        )

        eval_callback = EvalCallback(self.eval_env, best_model_save_path=self.work_dir,
                             log_path=self.work_dir, eval_freq=self.cfg.eval_frequency,
                             deterministic=True, render=False, n_eval_episodes=self.cfg.num_eval_episodes)

        self.agent.learn(self.cfg.num_train_steps,
                log_interval=10,
                callback=[eval_callback, checkpoint_callback],
                n_eval_episodes=self.cfg.num_eval_episodes,
                eval_log_path=self.work_dir,
                progress_bar=True
        )

        self.agent.save(os.path.join(self.work_dir, f"final_model.zip"))

        self.eval_env.close()
        self.train_env.close()


@hydra.main(config_path='configs/train.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()

