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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) 

jax.config.update("jax_platform_name", "cpu")

def is_valid_env(state, plan) -> bool:
    """
    Return False if this world/seed is known to be incompatible with the plan.
    Implement your own checks here (e.g., required blocks present/reachable).
    """
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Craftax training data with configurable logging levels",
    )
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
        default=5,
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for detailed debugging",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="ERROR",
        help="Set logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging based on command line arguments
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("astar").setLevel(logging.DEBUG)
        # Import and configure A* logging
        try:
            from astar import set_astar_verbose
            set_astar_verbose(True)
            logger.info("Verbose logging enabled for A* pathfinding")
        except ImportError:
            logger.warning("Could not import A* logging functions")
        logger.info("Verbose logging enabled")
    else:
        # Set the specified log level
        log_level = getattr(logging, args.log_level.upper())
        logging.getLogger().setLevel(log_level)
        logging.getLogger("astar").setLevel(log_level)
        # Configure A* logging to match
        try:
            from astar import set_astar_log_level
            set_astar_log_level(log_level)
            logger.info(f"A* logging level set to {args.log_level}")
        except ImportError:
            logger.warning("Could not import A* logging functions")
        logger.info(f"Logging level set to {args.log_level}")

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
            (bt.TREE, [at.DO], "wood"),
            (bt.TREE, [at.DO], "wood"),
            (bt.GRASS, [at.PLACE_TABLE], "table"),
            (bt.TREE, [at.DO], "wood"),
            (bt.CRAFTING_TABLE, [at.MAKE_WOOD_PICKAXE], "wooden_pickaxe"),
            (bt.STONE, [at.DO], "stone"),
            (bt.STONE, [at.DO], "stone"),
            (bt.STONE, [at.DO], "stone"),
            (bt.STONE, [at.DO], "stone"),
            (bt.STONE, [at.DO], "stone"),
            (bt.STONE, [at.DO], "stone"),
            (bt.STONE, [at.DO], "stone"),
            (bt.STONE, [at.DO], "stone"),
            (bt.TREE, [at.DO], "wood"),
            (bt.TREE, [at.DO], "wood"),
            (bt.CRAFTING_TABLE, [at.MAKE_STONE_PICKAXE], "stone_pickaxe"),
            # (bt.STONE, [at.DO], "stone"),
            # (bt.STONE, [at.DO], "stone"),
            # (bt.STONE, [at.DO], "stone"),
            (bt.COAL, [at.DO], "coal"),
            (bt.COAL, [at.DO], "coal"),
            # (bt.COAL, [at.DO], "coal"),
            # (bt.COAL, [at.DO], "coal"),
            # (bt.COAL, [at.DO], "coal"),
            (bt.IRON, [at.DO], "iron"),
            (bt.IRON, [at.DO], "iron"),
            (bt.PATH, [at.PLACE_FURNACE], "furnace"),
            # (bt.PATH, [at.PLACE_TABLE], "table"),
            (bt.FURNACE, [at.MAKE_IRON_PICKAXE], "iron_pickaxe"),

        ]
    ]

    inventory_goals = {
        # "wood": 2,
        "iron_pickaxe": 1,
        # "iron" : 2
    }

    all_task_data = []

    trace_nb = 0
    next_seed = args.base_seed
    attempts = 0

    while trace_nb < args.samples and attempts < args.max_attempts:
        seed = next_seed
        next_seed += 1
        attempts += 1

        logger.info(f"Attempt {attempts}/{args.max_attempts}: Generating trace {trace_nb + 1}/{args.samples} with seed {seed}")

        # Seed all RNGs deterministically per episode
        np.random.seed(seed)
        random.seed(seed)
        key = jax.random.PRNGKey(seed)

        # Choose a plan using the Python RNG (seeded above)
        plan = random.choice(plans)
        logger.info(f"Selected plan: {plan}")

        # Split per-episode key for env.reset and later for execute_plan
        key, key_reset = jax.random.split(key)
        # Obs are pixel obs; State is all game data
        logger.info("Resetting environment...")
        obs, state = env.reset(key_reset, env_params)
        logger.info(f"Environment reset complete. Player position: {state.player_position}")

        # Skip seeds with invalid worlds/configs for this plan
        if not is_valid_env(state, plan):
            logger.warning(f"Skipping seed {seed}: invalid world for selected plan")
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

        # Initialize truth for the initial observation (before any actions)
        all_truths.append(plan[0][2])

        try:
            logger.info(f"Executing plan with {len(plan)} steps...")
            for step_idx, (target, actions, truth) in enumerate(plan):
                logger.info(f"Step {step_idx + 1}/{len(plan)}: Target={target}, Actions={actions}, Truth={truth}")
                
                # New key per plan step to keep JAX RNG usage clean
                key, key_step = jax.random.split(key)
                state, obs_set, action_log, state_set, rew_set, info_set = execute_plan(
                    env, key_step, state, env_params, target, actions
                )

                logger.info(f"Step {step_idx + 1} completed: {len(action_log)} actions, {len(obs_set)} observations")

                all_states.extend(state_set)
                all_actions.extend(action_log)
                all_rewards.extend(rew_set)
                all_info.extend(info_set)

                # Assign truths so that each step labels only its own frames
                added = len(obs_set)
                all_truths.extend([truth] * added)

                if args.obs == "pixels":
                    for s, imgs in zip(state_set, obs_set):
                        all_obs.append(get_top_down_obs(s, imgs))
                else:
                    all_obs.extend(obs_set)

            # Pad terminal step metadata (actions/rewards/info); truths already aligned to observations
            all_actions.append(0)
            all_rewards.append(0)
            all_info.append(all_info[-1])

            inventory_state = [s.inventory for s in all_states]
            inventory_state_dicts = states_to_dicts(inventory_state)
            last_inventory = inventory_state_dicts[-1]

            valid_inv = True 

            for inv_req in inventory_goals:
                if last_inventory.get(inv_req, 0) < inventory_goals[inv_req]:
                    valid_inv = False
                    logger.warning(f"Trace failed inventory goal: needs {inventory_goals[inv_req]} {inv_req}, has {last_inventory.get(inv_req, 0)}")
            
            if valid_inv:
                logger.info(f"Trace met inventory goals: {last_inventory}")
            else:
                logger.warning(f"Trace did not meet inventory goals: {last_inventory}, skipping trace")
                continue
            

            all_truths = print_action_timeline(inventory_state)

            # Sanity check
            if len(all_obs) != len(all_truths):
                logger.warning(f"obs/truth length mismatch: obs={len(all_obs)} truths={len(all_truths)}")

            logger.info(f"Trace generation successful: {len(all_actions)} total actions, {len(all_obs)} observations")

            data = {
                "all_obs": all_obs,
                "all_states": all_states,
                "all_actions": all_actions,
                "all_truths": all_truths,
                "plan": plan,
                "seed": seed,
            }

        

            output_file = os.path.join(args.path, "raw_data", f"craftax_{trace_nb}.pkl")
            with open(output_file, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Trace saved to {output_file}")

            trace_nb += 1  # Only increment when we successfully generated a trace

            logger.info("Generating GIF visualization...")
            gen_gif(args, f"trace_{trace_nb}", all_obs, all_rewards, all_truths, all_actions)
            logger.info("GIF generation completed")
            
        except Exception as e:
            logger.error(f"Failed to generate trace with seed {seed}: {e}", exc_info=True)
            continue

    if trace_nb < args.samples:
        logger.warning(f"Generated {trace_nb}/{args.samples} traces before hitting max attempts ({attempts}).")
    else:
        logger.info(f"Successfully generated all {args.samples} traces in {attempts} attempts")

