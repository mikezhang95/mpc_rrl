# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import time
import json
import argparse
from typing import Optional

import numpy as np
import torch
import imageio

import utils
import hydra

from envs.carla.constants import *
from test_ppo import Tester

TEST_EPISODES = 1 # tests per model per perturb exp

class VideoRecorder(object):
    def __init__(self, height=480, width=480, camera_id=0, fps=30):
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def reset(self, ):
        self.frames = []

    def record(self, env):
        frames = env.render()
        self.frames.extend(frames)

    def save(self, video_path):
        imageio.mimsave(video_path, self.frames, fps=self.fps)


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

    tester.video_recorder = VideoRecorder(fps=FPS)
    tester.robust_test()




