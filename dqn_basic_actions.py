# dqn_craftax_sb3_gymnasium.py
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit  # (can remove if unused)

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.common.logger import configure
from top_down_env_gymnasium import CraftaxTopDownEnv

import imageio

if __name__ == "__main__":
    # Centralized kwargs for reuse


    env_kwargs = dict(
        render_mode="rgb_array",
        reward_items=["wood"],
        done_item="wood_pickaxe",
        include_base_reward=False,
    )

    vec_env = make_vec_env(
        CraftaxTopDownEnv,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs=env_kwargs,
        wrapper_kwargs={"max_steps": 100},
    )

    vec_env = VecTransposeImage(vec_env)

    policy_kwargs = dict(
        net_arch=[256, 256],
    )

    model = DQN(
        policy="CnnPolicy",
        env=vec_env,
        verbose=1,
    )

    model.learn(total_timesteps=500, progress_bar=True)

    model.save("dqn_craftax_topdown")
