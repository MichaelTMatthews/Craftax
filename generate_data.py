from craftax.craftax_env import make_craftax_env_from_name
import numpy as np
import os
import block_types as bt
import action_types as at
from helpers import *
import argparse
from tqdm import tqdm
import json
import jax
from collections import Counter
import imageio

jax.config.update("jax_platform_name", "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obs",
        type=str,
        required=False,
        default="pixels",
        help="symbolic or pixel observations",
    )
    parser.add_argument(
        "--samples",
        type=int,
        required=False,
        default=1,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=False,
        default="Traces/Test/",
        help="Path to save the generated traces",
    )

    args = parser.parse_args()

    if args.obs == "symbolic":
        env = make_craftax_env_from_name(
            "Craftax-Classic-Symbolic-v1", auto_reset=False
        )
    else:
        env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)

    os.makedirs(args.path, exist_ok=True)
    os.makedirs(args.path + "raw_data", exist_ok=True)

    env_params = env.default_params
    env_params = env.default_params.replace(
        max_timesteps=100000,  # Ensure long episodes
        day_length=99999,  # Make the day long
        mob_despawn_distance=0,  # Remove mob interactions
    )

    plans = [
        [
            (bt.TREE, [at.DO], "wood"),
            (bt.TREE, [at.DO], "wood"),
            (bt.GRASS, [at.PLACE_TABLE], "table"),
            (bt.TREE, [at.DO], "wood"),
            (bt.CRAFTING_TABLE, [at.MAKE_WOOD_PICKAXE], "wooden_pickaxe"),
            (bt.TREE, [at.DO], "wood"),
            (bt.STONE, [at.DO], "stone"),
            (bt.CRAFTING_TABLE, [at.MAKE_STONE_PICKAXE], "stone_pickaxe"),
        ]
    ]

    trace_nb = 0
    while trace_nb < args.samples:
        rng, rngs = reset_env(0)
        np.random.seed(0)

        # Obs are pixel obs
        # State is all game data
        obs, state = env.reset(rngs[0], env_params)

        all_obs = []
        all_states = []
        all_info = []
        all_rewards = []
        all_actions = []
        all_truths = ""

        if args.obs == "pixels":
            all_obs.append(get_top_down_obs(state, obs.copy()))
        else:
            all_obs.append(obs.copy())


        
        plan = plans[0]

        for target, actions, truth in plan:
            state, obs_set, action_log, state_set, rew_set, info_set = execute_plan(env, rngs[2], state, env_params, target, actions)

            all_states.extend(state_set)
            all_actions.extend(action_log)
            all_rewards.extend(rew_set)
            all_info.extend(info_set)
            all_truths += "\n".join([truth] * len(obs_set)) + "\n"

            if args.obs == "pixels":
                for states, imgs in zip(state_set, obs_set):
                    all_obs.append(get_top_down_obs(states, imgs))
            else:
                all_obs.extend(obs_set)

        

        import matplotlib.pyplot as plt

        gif_path = os.path.join(args.path, f"trace_{trace_nb}_obs.gif")
        frames = []

        # Ensure all_obs, all_rewards, all_truths, all_actions are aligned
        for idx, obs_img in enumerate(all_obs):
            fig, ax = plt.subplots(figsize=(4, 4))
            if obs_img.ndim == 3 and obs_img.shape[2] == 1:
                obs_img = obs_img.squeeze(-1)
            if obs_img.ndim == 2:  # grayscale to RGB
                obs_img = np.stack([obs_img]*3, axis=-1)
            ax.imshow(obs_img)
            ax.axis('off')

            # Get corresponding reward, truth, and action
            reward = all_rewards[idx] if idx < len(all_rewards) else ""
            truth = all_truths.split("\n")[idx] if idx < len(all_truths.split("\n")) else ""
            action = str(all_actions[idx]) if idx < len(all_actions) else ""

            text_str = f"Reward: {reward}\nTruth: {truth}\nAction: {action}"
            ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            fig.canvas.draw()

            # Logical size
            w, h = fig.canvas.get_width_height()

            # Raw RGBA bytes
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)

            # Pixels in buffer (per channel grouped)
            pixels = buf.size // 4
            logical_pixels = w * h

            # Compute HiDPI scale (usually 1 on non-Retina, 2 on Retina)
            scale = int(round((pixels / logical_pixels) ** 0.5)) or 1
            W, H = w * scale, h * scale

            frame = buf.reshape(H, W, 4)[..., :3]  # drop alpha to get RGB
            frames.append(frame)
            plt.close(fig)

        imageio.mimsave(gif_path, frames, duration=0.2)
        print(f"Saved gif to {gif_path}")

        trace_nb += 1
