import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor
from top_down_env_gymnasium import CraftaxTopDownEnv

import imageio

def to_gif_frame(obs):
    import numpy as np
    arr = np.asarray(obs)

    # remove vec batch dim if present (N=1)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]                    # (C,H,W)

    # CHW -> HWC if needed
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))   # (H,W,C)

    # if single channel, replicate to RGB
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)      # (H,W,3)

    # scale/clip & cast to uint8
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255)
        if arr.max() <= 1.0:
            arr = (arr * 255.0).round()
        arr = arr.astype(np.uint8)

    # if still grayscale 2D, OK for GIF; otherwise ensure HWC
    return arr


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

    model = PPO.load("ppo_craftax_wood_ppo_actions")

    # model = PPO(
    #     policy="CnnPolicy",
    #     env=train_env,
    #     verbose=1,
    #     tensorboard_log="./tb_logs_ppo_craftax",
    #     device="auto",

    # )

    # model.learn(
    #     total_timesteps=500_000,
    #     log_interval=10,
    #     tb_log_name="ppo_wood_actions",
    #     progress_bar=True,
    # )
    # model.save("ppo_craftax_wood_ppo_actions")

    # -------- Eval vec env (identical preprocessing & random-seed-per-episode) --------
    eval_env = DummyVecEnv([make_env(rng=np.random.default_rng(6887))])
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecMonitor(eval_env)

    obs = eval_env.reset()
    images = [to_gif_frame(obs)]

    done = False
    steps = 0
    while not done and steps < 100:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        images.append(to_gif_frame(obs))
        done = bool(dones[0])
        steps += 1

    imageio.mimsave("craftax_run_test_wood_easy_ppo_actions_2.gif", images, fps=5)