# %%
import jax
import jax.numpy as jnp
import pygame
import numpy as np
from random import randint

from craftax_marl.envs.craftax_pixels_env import CraftaxMARLPixelsEnv as CraftaxEnv
from craftax_marl.game_logic import craftax_step
from craftax_marl.renderer import render_craftax_pixels
from craftax_marl.constants import BLOCK_PIXEL_SIZE_HUMAN, OBS_DIM, INVENTORY_OBS_HEIGHT
from craftax_marl.util.game_logic_utils import *
from craftax_marl.util.maths_utils import *

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
    pygame.K_BACKSPACE: Action.REQUEST_FOOD,
    pygame.K_BACKSLASH: Action.REQUEST_DRINK,
    pygame.K_RETURN: Action.REQUEST_WOOD,
    pygame.K_RSHIFT: Action.REQUEST_STONE,
    pygame.K_UP: Action.REQUEST_IRON,
    pygame.K_DOWN: Action.REQUEST_COAL,
    pygame.K_LEFT: Action.REQUEST_DIAMOND,
    pygame.K_RIGHT: Action.GIVE,
}

class CraftaxRenderer:
    def __init__(self, env: CraftaxEnv, env_params, pixel_render_size=64//BLOCK_PIXEL_SIZE_HUMAN):
        self.env = env
        self.env_params = env_params
        self.pixel_render_size = pixel_render_size
        self.pygame_events = []

        self.screen_size = (
            OBS_DIM[1] * BLOCK_PIXEL_SIZE_HUMAN * pixel_render_size,
            (2 + OBS_DIM[0] + INVENTORY_OBS_HEIGHT)
            * BLOCK_PIXEL_SIZE_HUMAN
            * pixel_render_size,
        )

        # Init rendering
        pygame.init()
        pygame.key.set_repeat(250, 75)

        self.screen_surface = pygame.display.set_mode(self.screen_size)

        self._render = render_craftax_pixels

    def update(self):
        # Update pygame events
        self.pygame_events = list(pygame.event.get())

        # Update screen
        pygame.display.flip()

    def render(self, env_state):
        # Clear
        self.screen_surface.fill((0, 0, 0))

        pixels = self._render(env_state, BLOCK_PIXEL_SIZE_HUMAN, env.static_env_params, env.player_specific_textures)[0]
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=0)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=1)

        surface = pygame.surfarray.make_surface(np.array(pixels).transpose((1, 0, 2)))
        self.screen_surface.blit(surface, (0, 0))
        return pixels

    def is_quit_requested(self):
        for event in self.pygame_events:
            if event.type == pygame.QUIT:
                return True
        return False
    
    def register_press(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    key = event.key
                    if key in KEY_MAPPING:
                        return KEY_MAPPING[key].value
                    else:
                        return Action.NOOP.value

# %%
rng = jax.random.PRNGKey(0)
env = CraftaxEnv(CraftaxEnv.default_static_params())
renderer = CraftaxRenderer(env, env.default_params)

jitted_reset = env.reset
jitted_step = jax.jit(craftax_step, static_argnames=("static_params", ))
obs, state = jitted_reset(rng+2, env.default_params)
state = state.replace(
    inventory=state.inventory.replace(
        pickaxe=jnp.array([1,1,1,1])
    )
)


# %%
print("Ready to play!")
while True:
    if renderer.is_quit_requested():
        break

    actions = [renderer.register_press()]
    actions.extend([
        randint(1,4) for _ in range(env.static_env_params.player_count-1)
    ])
    actions = jnp.array(actions)

    rng, _rng = jax.random.split(rng)
    state, _ = jitted_step(
        _rng, state, actions, env.default_params, CraftaxEnv.default_static_params()
    )
    renderer.render(state)
    renderer.update()

# %%
