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
    env_kwargs = dict(
        render_mode="rgb_array",
        reward_items=[],
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
    vec_env = VecMonitor(vec_env)

    policy_kwargs = dict(
        net_arch=[256, 256],
    )

    # model = DQN(
    #     policy="CnnPolicy",
    #     env=vec_env,
    #     verbose=1,
    #     learning_rate=1e-4,
    #     buffer_size=50_000,          # 100k–1M depending on RAM
    #     learning_starts=50_000,       # warmup before updates
    #     batch_size=256,
    #     gamma=0.99,
    #     train_freq=(4, "step"),
    #     gradient_steps=1,
    #     target_update_interval=10_000,
    #     exploration_fraction=0.8,     # slow anneal over a long run
    #     exploration_initial_eps=1.0,
    #     exploration_final_eps=0.05,
    #     # optimize_memory_usage helps with large replay buffers
    #     optimize_memory_usage=True,
    #     policy_kwargs=policy_kwargs,
    #     tensorboard_log="./tb_logs_dqn_craftax",
    #     device="auto",
    # )

    # model.learn(total_timesteps=1_000_000, log_interval=10, tb_log_name="run1", progress_bar=True)


    model = DQN(
        policy="CnnPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=500,          # 100k–1M depending on RAM
        learning_starts=500,       # warmup before updates
        batch_size=256,
        gamma=0.99,
        train_freq=(4, "step"),
        gradient_steps=1,
        target_update_interval=100,
        exploration_fraction=0.8,     # slow anneal over a long run
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        # optimize_memory_usage helps with large replay buffers
        # optimize_memory_usage=True,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tb_logs_dqn_craftax",
        device="auto",
    )

    model.learn(total_timesteps=1000, log_interval=10, tb_log_name="run1", progress_bar=True)

    model.save("dqn_craftax_wood_pickaxe_sparse")

    eval_env = CraftaxTopDownEnv(**env_kwargs)
    obs, info = eval_env.reset()
    images = [obs.copy()]
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        images.append(obs.copy())
        done = terminated or truncated

    frames = [(np.clip(f, 0, 1) * 255).astype(np.uint8) for f in images]
    imageio.mimsave(f"craftax_run_test_wp_sparse.gif", frames, fps=5)


