# dqn_craftax_sb3_gymnasium.py
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit  # (can remove if unused)

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor

from top_down_env_gymnasium import CraftaxTopDownEnv

import imageio

if __name__ == "__main__":
    # Centralized kwargs for reuse
    env_kwargs = dict(
        render_mode="rgb_array",          # keep consistent for pixel observations
        reward_items=["wood", "stone"],
        done_item="stone_pickaxe",
        include_base_reward=False,
    )

    # Proper usage: pass the class, not an instance
    vec_env = make_vec_env(
        CraftaxTopDownEnv,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs=env_kwargs,
        wrapper_kwargs={"max_steps": 10000},
    )

    vec_env = VecTransposeImage(vec_env)
    vec_env = VecMonitor(vec_env)

    policy_kwargs = dict(
        net_arch=[256, 256],
    )

    model = DQN(
        policy="CnnPolicy",
        env=vec_env,
        learning_rate=1e-4,
        buffer_size=200_000,
        learning_starts=10_000,
        batch_size=128,
        tau=1.0,
        gamma=0.99,
        train_freq=(4, "step"),
        gradient_steps=1,
        target_update_interval=10_000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tb_logs_dqn_craftax",
        device="auto",
    )

    model.learn(total_timesteps=500_000, progress_bar=True)
    model.save("dqn_craftax_topdown")

    # Evaluation env (reuse same kwargs for consistency)
    eval_env = make_vec_env(
        CraftaxTopDownEnv,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs=env_kwargs,
        wrapper_kwargs={"max_steps": 10000},
    )
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecMonitor(eval_env)

    obs = eval_env.reset()

    all_obs = [obs.copy()]
    total_reward = 0.0

    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        all_obs.append(obs.copy())
        total_reward += rewards[0]
        if np.any(dones):
            obs = eval_env.reset()

        frames = [f for f in all_obs]
        imageio.mimsave(f"craftax_dqn_sp_{total_reward}.gif", frames, fps=5)