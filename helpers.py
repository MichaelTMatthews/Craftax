import jax
from craftax.craftax_env import make_craftax_env_from_name
# from craftax.wrappers import AutoResetEnvWrapper, BatchEnvWrapper
import gym
import jax.numpy as jnp
from gym import Wrapper
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from astar import plan_to_object, plan_to_object_with_mining
import os
import block_types as bt
import action_types as at
from dataclasses import dataclass, asdict
import math
from PIL import Image
import numpy as np
from PIL import Image
import imageio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reset_env(seed = None ):
    # Only generate a random seed when seed is explicitly None.
    # This ensures seed=0 is respected and yields deterministic runs.
    if seed is None:
        seed = int.from_bytes(os.urandom(4), 'big')  # 32-bit random integer
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    rngs = jax.random.split(_rng, 3)
    return rng, rngs


def execute_plan(env, rng, state, env_params, target, actions):

    obs_set = []
    action_set = []  # Track actions, rewards, and achievements
    state_set = []
    reward_set = []
    info_set = []
    
    # Get map and player position
    map_ = state.map 
    start_pos = tuple(state.player_position.tolist())
    
    logger.info(f"Executing plan: target={target}, start_pos={start_pos}, additional_actions={actions}")
    logger.info(f"Map dimensions: {map_.shape}, map bounds: [{map_.min()}, {map_.max()}]")

    # Plan path with mining capability
    logger.info("Attempting pathfinding with mining capability...")
    plan_trace = plan_to_object_with_mining(map_, start_pos, target)
    
    if plan_trace is None:
        # Fallback to original pathfinding if mining pathfinding fails
        logger.warning("Mining pathfinding failed, falling back to standard pathfinding...")
        plan_trace = plan_to_object(map_, start_pos, target)
    
    if plan_trace is None:
        logger.error(f"Both pathfinding methods failed for target {target} from position {start_pos}")
        raise Exception(f"Could not find path to target {target} from position {start_pos}")
    
    logger.info(f"Pathfinding successful: {len(plan_trace)} movement actions")
    plan_trace.extend(actions)  # Append additional actions
    logger.info(f"Total plan: {len(plan_trace)} actions ({len(plan_trace) - len(actions)} movement + {len(actions)} additional)")

    # Execute plan
    logger.info("Executing plan...")
    for i, ac in enumerate(plan_trace):
        logger.debug(f"Step {i+1}/{len(plan_trace)}: Action {ac}")
        obs, state, reward, done, info = env.step(rng, state, ac, env_params)

        state_set.append(state)
        obs_set.append(obs.copy())
        action_set.append(ac)
        reward_set.append(reward)
        info_set.append(info)
        
        if done:
            logger.warning(f"Episode ended early at step {i+1}")

    logger.info(f"Plan execution completed: {len(action_set)} actions executed")
    return state, obs_set, action_set, state_set, reward_set, info_set

def states_to_dicts(states):
    states_dict = []
    for i, c in enumerate(states):
        d = asdict(c)
        d_int = {key: int(val.item()) for key, val in d.items()}
        states_dict.append(d_int)
    
    return states_dict


def get_skill_changepoints(state_dict):
    change_points = []

    for i in range(1, len(state_dict)):
        previous = state_dict[i - 1]
        current = state_dict[i]
        
        # Check each item in the current inventory
        for item, current_value in current.items():
            previous_value = previous.get(item, 0)
            if current_value > previous_value:
                change_points.append((item, i))
    
    return change_points

def get_skills(change_points):
    lines = []     
    prev_total = 0 
    group_sum = 0  
    prev_key = None

    for key, value in change_points:
        if key == prev_key:
            count = value - prev_total
        else:
            count = value - prev_total
            
        for _ in range(count):
            lines.append(key)
        prev_total = value
        prev_key = key

    final_string = "\n".join(lines)
    return final_string

def plot_states_actions(all_images, all_actions, filename="craftax.pdf"):
    num_images = len(all_images)
    cols = min(num_images, 7)  # Limit to 7 columns for readability
    rows = math.ceil(num_images / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))

    # Flatten axes array if there's more than one row
    if rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, (image, action) in enumerate(zip(all_images, all_actions)):
        axes[i].imshow(image)
        axes[i].set_title(action, fontsize=20)
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.savefig(filename)


def get_top_down_obs(state, obs, scale_factor=0.5, inventory_scale_factor=3):
    map_array = np.array(state.map)
    player_position = np.array(state.player_position)
    player_direction = int(state.player_direction)

    height, width = map_array.shape
    tile_size = 16  # Each tile is 16x16
    output_image = Image.new("RGB", (width * tile_size, height * tile_size))

    # Draw the map
    for y in range(height):
        for x in range(width):
            block_type = map_array[y, x]
            if block_type in bt.block_images:
                tile = bt.block_images[block_type]
                output_image.paste(tile, (x * tile_size, y * tile_size))

    # Overlay player
    py, px = player_position
    if 0 <= px < width and 0 <= py < height:
        player_sprite = bt.player_images.get(player_direction, bt.player_images[0])
        output_image.paste(player_sprite, (px * tile_size, py * tile_size), mask=player_sprite)

    # Downscale the image while keeping pixel art sharp
    new_size = (int(output_image.width * scale_factor), int(output_image.height * scale_factor))

    top_down_raw = output_image.resize(new_size, Image.NEAREST)
    raw_obs = (np.array(obs) * 255).astype(np.uint8)

    tools_inv = Image.fromarray(raw_obs[-7:-1, :])
    health_blocks_inv = Image.fromarray(raw_obs[-14:-8, :])

    # Upscale both images 2x using NEAREST neighbor interpolation
    upscale_factor = inventory_scale_factor
    tools_inv = tools_inv.resize(
        (tools_inv.width * upscale_factor, tools_inv.height * upscale_factor),
        resample=Image.NEAREST,
    )
    health_blocks_inv = health_blocks_inv.resize(
        (health_blocks_inv.width * upscale_factor, health_blocks_inv.height * upscale_factor),
        resample=Image.NEAREST,
    )

    H, W = top_down_raw.height, top_down_raw.width
    bottom_H, bottom_W = tools_inv.height, tools_inv.width
    new_size = max(H + bottom_H, W)  # Ensure the final image is square
    canvas = Image.new("RGB", (new_size, new_size), (0, 0, 0))

    canvas.paste(top_down_raw, (0, 0))
    canvas.paste(health_blocks_inv, (0, H))
    canvas.paste(tools_inv, (bottom_W, H))

    final_combined_state = np.array(canvas)
    final_combined_state = final_combined_state.astype(np.float32) / 255.0

    return final_combined_state


def inventory_satisfies_plan(inventory, plan, expected_counts=None):
    """
    Checks if the given inventory has at least the expected count for each resource in the plan.

    Args:
        inventory (dict): The inventory state dictionary.
        plan (list): List of tuples (resource, actions) required by the plan.
        expected_counts (dict, optional): Mapping of resource names to the expected minimum count.
                                          If None, assumes each resource is expected at least once.
    
    Returns:
        bool: True if inventory satisfies the requirements; False otherwise.
    """
    # If no expected counts are provided, assume each resource must appear at least once.
    if expected_counts is None:
        expected_counts = {resource: 1 for resource, _ in plan}

    for resource, count in expected_counts.items():
        if inventory.get(resource, 0) < count:
            return False
    return True


def gen_gif(args, name, all_obs, all_rewards, all_truths, all_actions):
    gif_path = os.path.join(args.path, f"{name}.gif")
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
        truth = all_truths[idx] if idx < len(all_truths) else ""
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

    imageio.mimsave(gif_path, frames, duration=100)
    print(f"Saved gif to {gif_path}")