# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import time
import json
import argparse
from typing import Optional
from collections import deque
import importlib
from omegaconf import OmegaConf

import hydra
import stable_baselines3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import sb3_contrib

import numpy as np
import torch

import utils
import hydra

from train_ppo import make_env

TEST_EPISODES = 100 # tests per model per perturb exp

class Tester(object):
    def __init__(
        self, args,
        results_dir: str,
        agent_dir: Optional[str],
        num_steps: Optional[int] = None,
        num_episodes: Optional[int] = 1,
    ):

        self.num_episodes = num_episodes
        self.video_recorder = None

        # setup path
        self.args = args
        self.results_path = results_dir
        self.test_path = os.path.join(self.results_path, "test")
        self.video_path = os.path.join(self.results_path, "videos")
        os.makedirs(self.video_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)

        # setup rollout steps
        self.num_steps = num_steps

        # load cfg
        self.cfg = utils.load_hydra_cfg(self.results_path)
        utils.set_seed_everywhere(self.cfg.seed)
        
        # load env
        self.cfg.env.params.domain_random = False
        self._create_env()

        # load agent
        # - M: cannot initialize with hydra due to env
        # -    so that initialize handly
        # cfg.agent.params.env = self.train_env
        # self.agent = hydra.utils.instantiate(cfg.agent)
        if 'ppo' in self.cfg.agent.name:
            self.cfg.agent.params.env = "tmp_env"
            self.cfg.agent.params.tensorboard_log = ""
        else:
            self.cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
            self.cfg.agent.params.action_dim = self.env.action_space.shape[0]
            self.cfg.agent.params.action_range = [
                float(self.env.action_space.low.min()),
                float(self.env.action_space.high.max())
                ]

        # initialize aggent
        agent = OmegaConf.to_container(self.cfg.agent, resolve=True)
        # PPO params
        agent_params = agent['params'] 
        if 'ppo' in self.cfg.agent.name:
            agent_params['env'] = self.env

        # PPO class
        module_class = agent['class'].split('.')
        module_name = '.'.join(module_class[:-1])
        class_name = module_class[-1]
        agent_module = importlib.import_module(module_name)
        agent_class = getattr(agent_module, class_name)

        # initialize agent 
        self.agent = agent_class(**agent_params)
        # self.agent.load(os.path.join(agent_dir, "best_model"))
        self.agent.load(os.path.join(agent_dir, f"final_model"))


    def _create_env(self, perturb_spec=None):
        self.cfg.env.params.is_render = self.video_recorder is not None
        self.cfg.env.params.port = self.args.port
        if perturb_spec is not None:
            self.cfg.env.params.perturb_spec = perturb_spec 
        # recreate env
        self.env = utils.make_env(self.cfg)

    def _test(self, perturb_spec, test_name):

        # clean env
        self.env.close()

        # setup perturb to env 
        self._create_env(perturb_spec)

        # reset recorder as well
        if self.video_recorder:
            self.video_recorder.reset()

        start_time = time.time()
        episode_rewards, episode_lengths, start_ids = [], [], []
        episode_route_error, episode_goal_error, episode_vel_error = [], [], []
        episode_acc = []
        episode_model_loss = []
        transitions = {
            "obs": deque(maxlen=self.cfg.agent.params.replay_buffer_size),
            "control": deque(maxlen=self.cfg.agent.params.replay_buffer_size),
            "next_obs": deque(maxlen=self.cfg.agent.params.replay_buffer_size),
        }
        episode_times = []
        for ep_id in range(self.num_episodes):
            start_id = ep_id * 13
            obs = self.env.reset(start_id)
            # For LSTM
            last_hidden_states = None
            episode_starts = np.ones((1,), dtype=bool)
            # for statistics
            done, i, total_reward = False, 0.0, 0.0
            total_route_error, total_goal_error, total_vel_error = 0.0, 0.0, 0.0
            total_acc = 0.0

            episode_times.append(0.0)

            try:
                while not done:

                    t_tmp = time.time()

                    action, hidden_states = self.agent.predict(obs, state=last_hidden_states, episode_start=episode_starts, deterministic=True)
                    episode_times[-1] += time.time() - t_tmp

                    next_obs, reward, done, info = self.env.step(action)
                    control = info["control_action"]
                    # # M: predict acceleration
                    # if hasattr(self.agent, "system_model"):
                    #     control = info["control_action"]
                    #     device = self.agent.system_model.device
                    #     pred_obs = self.agent.system_model(utils.to_torch(np.expand_dims(obs, axis=0), device), utils.to_torch(np.expand_dims(control, axis=0), device), utils.to_torch(np.expand_dims(action, axis=0), device))
                    #     acc = utils.to_np(pred_obs)[0, 0] - obs[2]
                    #     total_acc += np.abs(acc)
                    # # --------------------------

                    total_reward += reward
                    
                    if self.agent.__class__.__name__ == 'OnlineEstimationAgent':
                        transitions['obs'].append(obs)
                        transitions['control'].append(control)
                        transitions['next_obs'].append(next_obs)
                    obs = next_obs
                    last_hidden_states = hidden_states
                    i += 1

                    if self.video_recorder:
                        self.video_recorder.record(self.env)
                    
                    total_route_error += info.get('route_error', 1000.0)
                    total_goal_error = info.get('goal_error', 1000.0)
                    total_vel_error += info.get('vel_error', 1000.0)
                    if self.num_steps and i == self.num_steps or done:
                        break

            except Exception as e:
                print(f"Environment ErrorMsg: {e}")

            if self.agent.__class__.__name__ == 'OnlineEstimationAgent':
                if ep_id % self.cfg.agent.params.update_frequency == 0:

                    t_tmp = time.time()
                    loss = self.agent.update_parameters(transitions)
                    episode_model_loss.append(loss)
                
                    episode_times[-1] += time.time() - t_tmp

            print(f'Episode Ends! total_reward:{total_reward} route_error:{total_route_error} goal_error:{total_goal_error} vel_error:{total_vel_error}')
            start_ids.append(start_id)
            episode_rewards.append(total_reward)
            episode_lengths.append(i)
            episode_route_error.append(total_route_error)
            episode_goal_error.append(total_goal_error)
            episode_vel_error.append(total_vel_error)
            episode_acc.append(total_acc)

        mean_reward, std_reward, min_reward = np.mean(episode_rewards), np.std(episode_rewards), np.min(episode_rewards)
        mean_route_error, std_route_error, min_route_error = np.mean(episode_route_error), np.std(episode_route_error), np.min(episode_route_error)
        mean_goal_error, std_goal_error, min_goal_error = np.mean(episode_goal_error), np.std(episode_goal_error), np.min(episode_goal_error)
        mean_vel_error, std_vel_error, min_vel_error = np.mean(episode_vel_error), np.std(episode_vel_error), np.min(episode_vel_error)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

        print("\n=== Test {} on {} Episodes ===".format(perturb_spec, len(episode_lengths)))
        print("episode_reward: {:.2f} +/- {:.2f}".format(mean_reward,std_reward))
        print("episode_route_error: {:.2f} +/- {:.2f}".format(mean_route_error,std_route_error))
        print("episode_goal_error: {:.2f} +/- {:.2f}".format(mean_goal_error,std_goal_error))
        print("episode_vel_error: {:.2f} +/- {:.2f}".format(mean_vel_error,std_vel_error))
        print("episode_length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))
        print("Cost {:.2f} seconds.\n".format(time.time()-start_time))
        print("episode_times: {:.2f} +/- {:.2f}".format(np.mean(episode_times), np.std(episode_times)))
        
        # save test results
        save_data = {"episode_rewards": episode_rewards,
                     "episode_lengths": episode_lengths,
                     "episode_route_error": episode_route_error,
                     "episode_acc": episode_acc,
                     "episode_goal_error": episode_goal_error,
                     "episode_vel_error": episode_vel_error,
                     "episode_model_loss": episode_model_loss,
                     "mean_reward": mean_reward,
                     "std_reward": std_reward,
                     "min_reward": min_reward,
                     "perturb_spec": perturb_spec,
                     "start_ids": start_ids}


        if self.video_recorder:
            # save videos
            text_path = os.path.join(self.video_path, test_name)
            with open(text_path, "w") as wf:
                json.dump(save_data, wf) 
            video_path = text_path.replace("json", "mp4")
            self.video_recorder.save(video_path)
            print(f"Videos saved to {video_path}")

        else:
            save_path = os.path.join(self.test_path, test_name)
            with open(save_path, "w") as wf:
                json.dump(save_data, wf) 
            print(f"Test Results saved to {save_path}")


    def robust_test(self):

        perturb_param = self.args.perturb_param
        perturb_values = self.env.task.get_test_range(perturb_param)
        if perturb_param == 'town_pose':
            perturb_param = 'town'

        for perturb_value in perturb_values:
            test_perturb_spec = dict()
            if perturb_param == 'town':
                perturb_value = perturb_value[0]
            test_perturb_spec[perturb_param] = perturb_value
            print(f"Testing: {test_perturb_spec}")
            test_name = f"{perturb_param}-{perturb_value}.json"
            self._test(test_perturb_spec, test_name)

        print("\n=== All Robust Test Finished! ===")
        self.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default=None,
        help="The directory where the original experiment was run.",
    )
    parser.add_argument(
        "--agent_dir",
        type=str,
        default=None,
        help="The directory where the agent configuration and data is stored. "
        "If not provided, a mpc agent will be used.",
    )
    parser.add_argument(
        "--perturb_param",
        type=str,
        default=None,
        help="The directory where the agent configuration and data is stored. "
        "If not provided, a mpc agent will be used.",
    )
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    tester = Tester(
        args,
        results_dir=args.experiments_dir,
        agent_dir=args.agent_dir,
        num_steps=args.num_steps,
        num_episodes=TEST_EPISODES
    )
    tester.robust_test()




