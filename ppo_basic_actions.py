import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor
from top_down_env_gymnasium import CraftaxTopDownEnv

import imageio

# ---- Wrapper: draw a new RNG seed every episode ----
class RandomSeedOnReset(gym.Wrapper):
    def __init__(self, env, rng=None):
        super().__init__(env)
        self.rng = np.random.default_rng() if rng is None else rng

    # Gymnasium-style signature helps readability, but **kwargs also works
    def reset(self, *, seed=None, options=None, **kwargs):
        # Always override any incoming seed from VecEnv
        kwargs.pop("seed", None)
        new_seed = int(self.rng.integers(0, 2**31 - 1))
        return self.env.reset(seed=new_seed, options=options, **kwargs)

def make_env(rng=None):
    def _thunk():
        env = CraftaxTopDownEnv(
            render_mode=None,
            reward_items=[],
            done_item="wood",
            include_base_reward=False,
        )
        env = TimeLimit(env, max_episode_steps=25)
        env = RandomSeedOnReset(env, rng=rng)  # different seed every reset
        return env
    return _thunk

if __name__ == "__main__":
    base_rng = np.random.default_rng(12345)

    train_env = DummyVecEnv([make_env(rng=base_rng)])
    train_env = VecTransposeImage(train_env)  # HWC -> CHW for CnnPolicy
    train_env = VecMonitor(train_env)

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log="./tb_logs_ppo_craftax",
        device="auto",

    )

    model.learn(
        total_timesteps=500_000,
        log_interval=10,
        tb_log_name="ppo_wood_actions",
        progress_bar=True,
    )
    model.save("ppo_craftax_wood_ppo_actions")

    # -------- Eval vec env (identical preprocessing & random-seed-per-episode) --------
    eval_env = DummyVecEnv([make_env(rng=np.random.default_rng(9876))])
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecMonitor(eval_env)

    obs, info = eval_env.reset()
    images = [obs.copy()]
    done = False
    steps = 0

    # Roll out up to 100 steps, capturing true RGB frames for the GIF
    while not done and steps < 100:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        images.append(obs.copy())
        done = bool(terminated[0] or truncated[0])
        steps += 1
        print(f"Step {steps} Action {action} Reward {reward} Done {done}")

    imageio.mimsave("craftax_run_test_wood_easy_ppo_actions.gif", images, fps=5)