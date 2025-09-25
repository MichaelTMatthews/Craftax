# dqn_craftax_sb3_gymnasium_to_ppo.py
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit  # (can remove if unused)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.common.logger import configure
from top_down_env_gymnasium import CraftaxTopDownEnv

import imageio

if __name__ == "__main__":
    env_kwargs = dict(
        render_mode="rgb_array",
        reward_items=["wood", "stone", "wood_pickaxe"],
        done_item="stone_pickaxe",
        include_base_reward=False,
    )

    vec_env = make_vec_env(
        CraftaxTopDownEnv,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs=env_kwargs,
        wrapper_kwargs={"max_steps": 50},
    )

    # PPO with CnnPolicy expects channel-first; transpose HWC -> CHW
    vec_env = VecTransposeImage(vec_env)
    vec_env = VecMonitor(vec_env)

    policy_kwargs = dict(
        net_arch=[256, 256],
    )

    # Key PPO hyperparameters:
    # - n_steps * n_envs should be >= batch_size and usually divisible by it
    # - no buffer_size, learning_starts, train_freq, or exploration params
    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        verbose=1,
        # learning_rate=3e-4,
        # n_steps=512,           # rollout length per env
        # batch_size=256,         # must be <= n_steps * n_envs
        # n_epochs=10,            # minibatch passes per update
        # gamma=0.99,
        # gae_lambda=0.95,
        # clip_range=0.2,
        # ent_coef=0.0,
        # vf_coef=0.5,
        # max_grad_norm=0.5,
        # policy_kwargs=policy_kwargs,
        tensorboard_log="./tb_logs_ppo_craftax",
        device="auto",
    )

    model.learn(total_timesteps=1_000_000, log_interval=10, tb_log_name="run1_ppo", progress_bar=True)

    model.save("ppo_craftax_wood_pickaxe_sparse")

    # --- Quick deterministic rollout & GIF ---
    eval_env = CraftaxTopDownEnv(**env_kwargs)
    obs, info = eval_env.reset()
    images = [obs.copy()]
    done = False
    steps = 0
    while not done and steps < 200:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        images.append(obs.copy())
        done = terminated or truncated
        steps += 1
        print(f"Step {steps} Action {action} Reward {reward} Done {done}")

    imageio.mimsave("craftax_run_test_wp_sparse_ppo.gif", images, fps=5)