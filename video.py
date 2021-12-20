import json
from time import time

import gym
import imageio
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.vec_env import VecVideoRecorder
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv
from stable_baselines3.common.atari_wrappers import FireResetEnv
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.ppo import CnnPolicy

from benchmarks.activation import ActivationFunction
from benchmarks.networks import LayerConfig
from benchmarks.networks import VariableBenchmark
from benchmarks.settings import *
from benchmarks.callbacks import TimeLimitCallback
from benchmarks.wrapper import AtariWrapper


def main():
    set_random_seed(RANDOM_SEED)

    env_args = dict(
        frame_skip=1,
        screen_size=84,
        terminal_on_life_loss=True,
        clip_reward=True,
    )

    def atari_wrapper(env: gym.Env) -> gym.Env:
        env = AtariWrapper(env, **env_args)
        return env

    def make_env(rank: int, count: int) -> VecEnv:
        return make_vec_env(
            ENV_NAME,
            n_envs=count,
            seed=RANDOM_SEED + rank,
            start_index=0,
            monitor_dir=None,
            wrapper_class=atari_wrapper,
            env_kwargs=None,
            vec_env_cls=None,
            vec_env_kwargs=None,
            monitor_kwargs=None,
        )

    set_random_seed(RANDOM_SEED)
    env = make_env(1, 1)
    model = PPO.load(
        "checkpoints/PPO/Pong-v0/BestFromTrials512/model.zip",
        policy_kwargs=dict(
            features_extractor_class=VariableBenchmark,
            features_extractor_kwargs=dict(
                layers=[
                    LayerConfig(128, 2, 1, 0, ActivationFunction.GELU),
                    LayerConfig(16, 2, 1, 0, ActivationFunction.RELU),
                    LayerConfig(32, 8, 1, 0, ActivationFunction.CELU),
                ]
            ),
        ),
        env=env,
    )

    images = []
    obs = model.env.reset()
    img = model.env.render(mode="rgb_array")
    for i in range(1_000):
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, _, _ = model.env.step(action)
        img = model.env.render(mode="rgb_array")

    imageio.mimsave("E:\\RL Search Video\\BestFromTrials512.gif", [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=29.97)

    set_random_seed(RANDOM_SEED)
    env = make_env(1, 1)
    model = PPO.load("checkpoints/PPO/Pong-v0/NatureCNN/model.zip", policy_kwargs=dict(features_extractor_class=NatureCNN), env=env)

    images = []
    obs = model.env.reset()
    img = model.env.render(mode="rgb_array")
    for i in range(1_000):
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, _, _ = model.env.step(action)
        img = model.env.render(mode="rgb_array")

    imageio.mimsave("E:\\RL Search Video\\NatureCNN.gif", [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=29.97)

    # set_random_seed(RANDOM_SEED)
    # model = PPO.load("checkpoints/PPO/Pong-v0/BestFromTrials/model.zip", policy_kwargs=dict(features_extractor_class=FeatureExtractor))
    # reward_mean, reward_std = evaluate_policy(model, make_env(2, 1))
    # print(f"Best From Trials: {reward_mean}, {reward_std:0.4f}")


if __name__ == "__main__":
    main()
