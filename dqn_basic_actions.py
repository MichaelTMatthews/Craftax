# dqn_craftax_sb3_gymnasium.py
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor

from top_down_env_gymnasium import CraftaxTopDownEnv

if __name__ == "__main__":

    # 1) Build a Gymnasium VecEnv from your Gymnasium env class
    vec_env = make_vec_env(
        CraftaxTopDownEnv,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs={"render_mode": None},
        wrapper_kwargs={"max_steps": 10000},
    )

    # 2) Channel-first for CNN policies (HWC -> CHW)
    vec_env = VecTransposeImage(vec_env)

    # 3) Episode stats logging at VecEnv level
    vec_env = VecMonitor(vec_env)

    # 4) DQN config tuned for pixels (conservative baseline)
    policy_kwargs = dict(
        net_arch=[256, 256],  # MLP after CNN feature extractor
        # Optionally: features_extractor_class / kwargs for a custom CNN
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

    # 5) Train
    model.learn(total_timesteps=500_000, progress_bar=True)
    model.save("dqn_craftax_topdown")

    # 6) Quick sanity rollout (SB3 VecEnv: step() -> (obs, rewards, dones, infos))
    eval_env = make_vec_env(
        CraftaxTopDownEnv,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs={"render_mode": None},
        wrapper_kwargs={"max_steps": 10000},
    )
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecMonitor(eval_env)

    obs = eval_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        if np.any(dones):
            obs = eval_env.reset()