# %%
from craftax_marl.constants import *
import jax.numpy as jnp
import jax
import numpy as np
from PIL import Image
from craftax_marl.envs.craftax_pixels_env import CraftaxMARLPixelsEnv as CraftaxEnv
from craftax_marl.world_gen.world_gen import generate_world

# %%
rng = jax.random.PRNGKey(0)
env = CraftaxEnv(CraftaxEnv.default_static_params())
state = generate_world(rng, env.default_params, env.static_env_params)
obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)
static_params = env.default_static_params()


# %%
from craftax_marl.renderer import render_craftax_pixels

obs = render_craftax_pixels(
    state,
    BLOCK_PIXEL_SIZE_HUMAN,
    static_params,
    env.player_specific_textures
)

# %%