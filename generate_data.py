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
import matplotlib.pyplot as plt
import random 
import pickle 

jax.config.update("jax_platform_name", "cpu")

def is_valid_env(state, plan) -> bool:
    """
    Return False if this world/seed is known to be incompatible with the plan.
    Implement your own checks here (e.g., required blocks present/reachable).
    """
    return True

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
        default=10,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=False,
        default="Traces/Test/",
        help="Path to save the generated traces",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        required=False,
        default=0,
        help="Base seed used to derive per-episode seeds",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        required=False,
        default=1000,
        help="Upper bound on total seed attempts to reach the requested samples",
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
        ],
        [
            (bt.TREE, [at.DO], "wood"),
            (bt.STONE, [at.DO], "stone"),
        ]
    ]

    all_task_data = []

    trace_nb = 0
    next_seed = args.base_seed
    attempts = 0

    while trace_nb < args.samples and attempts < args.max_attempts:
        seed = next_seed
        next_seed += 1
        attempts += 1

        # Seed all RNGs deterministically per episode
        np.random.seed(seed)
        random.seed(seed)
        key = jax.random.PRNGKey(seed)

        # Choose a plan using the Python RNG (seeded above)
        plan = random.choice(plans)

        # Split per-episode key for env.reset and later for execute_plan
        key, key_reset = jax.random.split(key)
        # Obs are pixel obs; State is all game data
        obs, state = env.reset(key_reset, env_params)

        # Skip seeds with invalid worlds/configs for this plan
        if not is_valid_env(state, plan):
            print(f"Skipping seed {seed}: invalid world for selected plan")
            continue

        all_obs = []
        all_states = [state]
        all_info = []
        all_rewards = []
        all_actions = []
        all_truths = []

        if args.obs == "pixels":
            all_obs.append(get_top_down_obs(state, obs.copy()))
        else:
            all_obs.append(obs.copy())

        try:
            for target, actions, truth in plan:
                # New key per plan step to keep JAX RNG usage clean
                key, key_step = jax.random.split(key)
                state, obs_set, action_log, state_set, rew_set, info_set = execute_plan(
                    env, key_step, state, env_params, target, actions
                )

                all_states.extend(state_set)
                all_actions.extend(action_log)
                all_rewards.extend(rew_set)
                all_info.extend(info_set)
                all_truths.extend([truth] * len(obs_set))

                if args.obs == "pixels":
                    for s, imgs in zip(state_set, obs_set):
                        all_obs.append(get_top_down_obs(s, imgs))
                else:
                    all_obs.extend(obs_set)

            # Pad terminal step metadata
            all_actions.append(0)
            all_rewards.append(0)
            all_info.append(all_info[-1])
            all_truths.append(all_truths[-1])

            data = {
                "all_obs": all_obs,
                "all_states": all_states,
                "all_info": all_info,
                "all_rewards": all_rewards,
                "all_actions": all_actions,
                "all_truths": all_truths,
                "plan": plan,
                "seed": seed,
            }

            with open(os.path.join(args.path, "raw_data", f"craftax_{trace_nb}.pkl"), "wb") as f:
                pickle.dump(data, f)

            trace_nb += 1  # Only increment when we successfully generated a trace

            gen_gif(args, f"trace_{trace_nb}", all_obs, all_rewards, all_truths, all_actions)
        except Exception as e:
            print(f"Failed to generate trace with seed {seed}: {e}")
            continue

    if trace_nb < args.samples:
        print(f"Generated {trace_nb}/{args.samples} traces before hitting max attempts ({attempts}).")

