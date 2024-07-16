import bz2
import pickle
import sys
import time
from pathlib import Path

import argparse
from typing import Any

import pygame

import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax.constants import (
    OBS_DIM,
    BLOCK_PIXEL_SIZE_HUMAN,
    INVENTORY_OBS_HEIGHT,
    Action,
    Achievement,
)
from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv as CraftaxEnv
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.renderer import render_craftax_pixels

KEY_MAPPING = {
    pygame.K_q: Action.NOOP,
    pygame.K_w: Action.UP,
    pygame.K_d: Action.RIGHT,
    pygame.K_s: Action.DOWN,
    pygame.K_a: Action.LEFT,
    pygame.K_SPACE: Action.DO,
    pygame.K_1: Action.MAKE_WOOD_PICKAXE,
    pygame.K_2: Action.MAKE_STONE_PICKAXE,
    pygame.K_3: Action.MAKE_IRON_PICKAXE,
    pygame.K_4: Action.MAKE_DIAMOND_PICKAXE,
    pygame.K_5: Action.MAKE_WOOD_SWORD,
    pygame.K_6: Action.MAKE_STONE_SWORD,
    pygame.K_7: Action.MAKE_IRON_SWORD,
    pygame.K_8: Action.MAKE_DIAMOND_SWORD,
    pygame.K_t: Action.PLACE_TABLE,
    pygame.K_TAB: Action.SLEEP,
    pygame.K_r: Action.PLACE_STONE,
    pygame.K_f: Action.PLACE_FURNACE,
    pygame.K_p: Action.PLACE_PLANT,
    pygame.K_e: Action.REST,
    pygame.K_COMMA: Action.ASCEND,
    pygame.K_PERIOD: Action.DESCEND,
    pygame.K_y: Action.MAKE_IRON_ARMOUR,
    pygame.K_u: Action.MAKE_DIAMOND_ARMOUR,
    pygame.K_i: Action.SHOOT_ARROW,
    pygame.K_o: Action.MAKE_ARROW,
    pygame.K_g: Action.CAST_FIREBALL,
    pygame.K_h: Action.CAST_ICEBALL,
    pygame.K_j: Action.PLACE_TORCH,
    pygame.K_z: Action.DRINK_POTION_RED,
    pygame.K_x: Action.DRINK_POTION_GREEN,
    pygame.K_c: Action.DRINK_POTION_BLUE,
    pygame.K_v: Action.DRINK_POTION_PINK,
    pygame.K_b: Action.DRINK_POTION_CYAN,
    pygame.K_n: Action.DRINK_POTION_YELLOW,
    pygame.K_m: Action.READ_BOOK,
    pygame.K_k: Action.ENCHANT_SWORD,
    pygame.K_l: Action.ENCHANT_ARMOUR,
    pygame.K_LEFTBRACKET: Action.MAKE_TORCH,
    pygame.K_RIGHTBRACKET: Action.LEVEL_UP_DEXTERITY,
    pygame.K_MINUS: Action.LEVEL_UP_STRENGTH,
    pygame.K_EQUALS: Action.LEVEL_UP_INTELLIGENCE,
    pygame.K_SEMICOLON: Action.ENCHANT_BOW,
}


def save_compressed_pickle(title: str, data: Any):
    with bz2.BZ2File(title + ".pbz2", "w") as f:
        pickle.dump(data, f)


class CraftaxRenderer:
    def __init__(self, env: CraftaxEnv, env_params, pixel_render_size=4):
        self.env = env
        self.env_params = env_params
        self.pixel_render_size = pixel_render_size
        self.pygame_events = []

        self.screen_size = (
            OBS_DIM[1] * BLOCK_PIXEL_SIZE_HUMAN * pixel_render_size,
            (OBS_DIM[0] + INVENTORY_OBS_HEIGHT)
            * BLOCK_PIXEL_SIZE_HUMAN
            * pixel_render_size,
        )

        # Init rendering
        pygame.init()
        pygame.key.set_repeat(250, 75)

        self.screen_surface = pygame.display.set_mode(self.screen_size)

        self._render = jax.jit(render_craftax_pixels, static_argnums=(1,))

    def update(self):
        # Update pygame events
        self.pygame_events = list(pygame.event.get())

        # Update screen
        pygame.display.flip()

    def render(self, env_state):
        # Clear
        self.screen_surface.fill((0, 0, 0))

        pixels = self._render(env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=0)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=1)

        surface = pygame.surfarray.make_surface(np.array(pixels).transpose((1, 0, 2)))
        self.screen_surface.blit(surface, (0, 0))

    def is_quit_requested(self):
        for event in self.pygame_events:
            if event.type == pygame.QUIT:
                return True
        return False

    def get_action_from_keypress(self, state):
        if state.is_sleeping or state.is_resting:
            return Action.NOOP.value

        for event in self.pygame_events:
            if event.type == pygame.KEYDOWN:
                if event.key in KEY_MAPPING:
                    return KEY_MAPPING[event.key].value

        return None


def print_new_achievements(old_achievements, new_achievements):
    for i in range(len(old_achievements)):
        if old_achievements[i] == 0 and new_achievements[i] == 1:
            print(
                f"{Achievement(i).name} ({new_achievements.sum()}/{len(Achievement)})"
            )


def main(args):
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params

    print("Controls")
    for k, v in KEY_MAPPING.items():
        print(f"{pygame.key.name(k)}: {v.name.lower()}")

    if args.god_mode:
        env_params = env_params.replace(god_mode=True)

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset(_rng, env_params)

    pixel_render_size = 64 // BLOCK_PIXEL_SIZE_HUMAN

    renderer = CraftaxRenderer(env, env_params, pixel_render_size=pixel_render_size)
    renderer.render(env_state)

    step_fn = jax.jit(env.step)

    traj_history = {"state": [env_state], "action": [], "reward": [], "done": []}

    clock = pygame.time.Clock()

    while not renderer.is_quit_requested():
        action = renderer.get_action_from_keypress(env_state)

        if action is not None:
            rng, _rng = jax.random.split(rng)
            old_achievements = env_state.achievements
            obs, env_state, reward, done, info = step_fn(
                _rng, env_state, action, env_params
            )
            new_achievements = env_state.achievements
            print_new_achievements(old_achievements, new_achievements)

            if reward > 0.8:
                print(f"Reward: {reward}\n")

            traj_history["state"].append(env_state)
            traj_history["action"].append(action)
            traj_history["reward"].append(reward)
            traj_history["done"].append(done)

            renderer.render(env_state)

        renderer.update()
        clock.tick(args.fps)

    if args.save_trajectories:
        save_name = f"play_data/trajectories_{int(time.time())}"
        if args.god_mode:
            save_name += "_GM"
        save_name += ".pkl"
        Path("play_data").mkdir(parents=True, exist_ok=True)
        save_compressed_pickle(save_name, traj_history)


def entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument("--god_mode", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_trajectories", action="store_true")
    parser.add_argument("--fps", type=int, default=60)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)


if __name__ == "__main__":
    entry_point()
