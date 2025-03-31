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

jax.config.update('jax_platform_name', 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs', type=str, required=False, default="symbolic", help="symbolic or pixel observations")
    parser.add_argument('--samples', type=int, required=False, default=1, help="Number of samples to generate")
    parser.add_argument('--path', type=str, required=False, default="Traces/Test/", help="Path to save the generated traces")
    parser.add_argument('--obj', type=str, required=False, default="Build Stone Pickaxe", help="Objective to achieve")
    parser.add_argument('--shuffle', type=bool, required=False, default=False, help="Shuffle the plans")

    args = parser.parse_args()

    if args.obs == "symbolic":
        env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
        os.makedirs(args.path + 'symbolic_obs', exist_ok=True)
    else:
        env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
        os.makedirs(args.path + 'top_down_obs', exist_ok=True)
        os.makedirs(args.path + 'pixel_obs', exist_ok=True)

    os.makedirs(args.path, exist_ok=True)
    os.makedirs(args.path + 'groundTruth', exist_ok=True)
    os.makedirs(args.path + 'mapping', exist_ok=True)
    os.makedirs(args.path + 'actions', exist_ok=True)


    env_params = env.default_params
    env_params = env.default_params.replace(
        max_timesteps=100000,  # Ensure long episodes
        day_length=99999,  # Make the day long
        mob_despawn_distance=0,  # Remove mob interactions
    )
    

    plans = [
        (bt.TREE, [at.DO], "wood"), 
        (bt.TREE, [at.DO], "wood"), 

        (bt.GRASS, [at.PLACE_TABLE], "table"),
        (bt.TREE, [at.DO], "wood"),
        (bt.CRAFTING_TABLE, [at.MAKE_WOOD_PICKAXE], "wooden_pickaxe"),
        (bt.TREE, [at.DO], "wood"),
        (bt.STONE, [at.DO], "stone"),
        (bt.CRAFTING_TABLE, [at.MAKE_STONE_PICKAXE], "stone_pickaxe"),
    ]

    progress_bar = tqdm(total=args.samples)
    trace_nb = 0
    episode_lengths = []
    ground_truth_distribution = Counter()
    ground_truth_skills = set()

    #Seed numpy random generator
    np.random.seed(0)

    while trace_nb < args.samples:
        rng, rngs = reset_env()
        obs, state = env.reset(rngs[0], env_params)

        if args.shuffle:
            np.random.shuffle(plans)

        all_images = [obs.copy()]
        all_action_logs = []
        all_states = [state]
        all_truths = ""

        if args.obs == "pixels":
            all_top_down_states = [get_top_down_obs(state, obs.copy())]

        try: 
            for target, actions, truth in plans:
                state, images, action_log, state_set = execute_plan(env, rngs[2], state, env_params, target, actions)

                all_images.extend(images)
                all_action_logs.extend(action_log)
                all_states.extend(state_set)

                all_truths += "\n".join([truth] * len(images))
                all_truths += "\n"

                if args.obs == "pixels":
                    for states, imgs in zip(state_set, images):
                        all_top_down_states.append(get_top_down_obs(states, imgs))

            all_action_logs.append(0) #Add a NOOP action at the end of the trace
            
            inventory_state = [s.inventory for s in all_states]
            inventory_state_dicts = states_to_dicts(inventory_state)

            #Ensure the last inventory state dict has 1 stone, 1 coal and 1 wood
            last_inventory = inventory_state_dicts[-1]
            if last_inventory.get("stone_pickaxe", 0) < 1:
                print("Skipping trace, last inventory state does not meet req.")
                continue


            # Track episode length
            episode_length = len(all_images)
            episode_lengths.append(episode_length)
            
            # Track ground truth distribution
            for skill in all_truths.split("\n"):
                if skill:
                    ground_truth_distribution[skill] += 1
                    ground_truth_skills.add(skill)

            
            all_images = np.array(all_images)   
            all_actions_logs = np.array(all_action_logs)
            

            np.save(args.path + 'actions/craftax_' + str(trace_nb) + '.npy', all_actions_logs)
            with open(args.path + f"groundTruth/craftax_{trace_nb}", "w") as f:
                f.write(all_truths.rstrip("\n"))

            if args.obs == "pixels":
                all_top_down_states = np.array(all_top_down_states)
                np.save(args.path + 'top_down_obs/craftax_' + str(trace_nb) + '.npy', all_top_down_states)
                np.save(args.path + 'pixel_obs/craftax_' + str(trace_nb) + '.npy', all_images)
            
            else:
                np.save(args.path + 'symbolic_obs/craftax_' + str(trace_nb) + '.npy', all_images)
            
            trace_nb += 1
            progress_bar.update(1)
        except Exception as e:
            continue

    formatted_plans = [
        {"target": str(bt.block_translation_dict[target]), "actions": [str(at.action_translation_dict[action]) for action in actions],
         "skill": truth}
        for target, actions, truth in plans
    ]

    stats = {
        "min_episode_length": min(episode_lengths) if episode_lengths else 0,
        "avg_episode_length": sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0,
        "max_episode_length": max(episode_lengths) if episode_lengths else 0,
        "ground_truth_distribution": dict(ground_truth_distribution)
    }

    with open(os.path.join(args.path, "trace_config.json"), "w") as f:
        json.dump(
            {
                "parameters": vars(args),
                "plans": formatted_plans,
                "stats": stats,
            },
            f,
            indent=4,
        )
    
    #In the mapping file, save a mapping of the ground truth with a number to the left : 0 skill_1
    with open(args.path + "mapping/mapping.txt", "w") as f:
        for i, skill in enumerate(ground_truth_skills):
            f.write(f"{i} {skill}\n")

    print("Done.")
