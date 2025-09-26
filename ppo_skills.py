# train_maskable_ppo.py
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker  # <-- ensures masks used in rollout

from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor

from top_down_env_gymnasium import CraftaxTopDownEnv
from top_down_env_gymnasium_options import OptionsOnTopEnv

import imageio


class RandomSeedOnReset(gym.Wrapper):
    def __init__(self, env, rng=None):
        super().__init__(env)
        self.rng = np.random.default_rng() if rng is None else rng

    def reset(self, *, seed=None, options=None, **kwargs):
        kwargs.pop("seed", None)
        new_seed = int(self.rng.integers(0, 2**31 - 1))
        return self.env.reset(seed=new_seed, options=options, **kwargs)

    # NEW: forward masks so ActionMasker can see them
    def action_masks(self):
        return self.env.action_masks()


def mask_fn(env):
    return env.action_masks()


def make_options_env(*, rng, render_mode=None, K=5, max_episode_steps=25):
    def _thunk():
        base_env = CraftaxTopDownEnv(
            render_mode=render_mode,  # None for train; "rgb_array" for eval/gif
            reward_items=[],
            done_item="wood",
            include_base_reward=False,
            return_uint8=True,        # keep 274x274x3 uint8 obs
        )
        env = OptionsOnTopEnv(
            base_env,
            num_primitives=16,
            num_options=K,
            gamma=1,
            max_skill_len=50,
        )

        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = RandomSeedOnReset(env, rng=rng)
        env = ActionMasker(env, mask_fn)
        env = RecordEpisodeStatistics(env)
        return env
    return _thunk


if __name__ == "__main__":
    K = 5
    base_rng = np.random.default_rng(12345)

    train_env = DummyVecEnv([make_options_env(rng=base_rng, render_mode=None, K=K)])
    train_env = VecTransposeImage(train_env) 
    train_env = VecMonitor(train_env)

    model = MaskablePPO(
        "CnnPolicy",                   # pixels -> CNN
        train_env,
        verbose=1,
        tensorboard_log="./tb_logs_ppo_craftax",
    )


    eval_env_vec = DummyVecEnv([make_options_env(
        rng=np.random.default_rng(9876),
        render_mode=None,  
        K=K
    )])
    eval_env_vec = VecTransposeImage(eval_env_vec)
    eval_env_vec = VecMonitor(eval_env_vec)

    eval_cb = MaskableEvalCallback(
        eval_env_vec,
        eval_freq=10,
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(
        total_timesteps=500_000,
        tb_log_name="ppo_wood_options",   # TB subdir
        log_interval=10,
        progress_bar=True,
        callback=eval_cb,
    )
    obs, info = eval_env_vec.reset()
    frames = [obs.copy()]

    done = False
    steps = 0
    while not done and steps < 100:
        # Pull masks from the FIRST sub-env (vectorized)
        masks = eval_env_vec.envs[0].action_masks()
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env_vec.step(action)
        frames.append(obs.copy())
        done = bool(terminated[0] or truncated[0])
        steps += 1

    imageio.mimsave("craftax_ppo_options_eval.gif", frames, fps=5)

    last_info = info[0] if isinstance(info, (list, tuple)) else info
    print("Eval done.")
    print("Last episode reward:", last_info.get("episode", {}).get("r", None))