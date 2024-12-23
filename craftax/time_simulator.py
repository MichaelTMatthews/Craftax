# %%
import jax
import jax.numpy as jnp

rng = jax.random.PRNGKey(0)

# %%
from craftax_marl.envs.craftax_pixels_env import CraftaxMARLPixelsEnv as CraftaxEnv

env = CraftaxEnv(CraftaxEnv.default_static_params())

# %%
# Generate World
from craftax_marl.world_gen.world_gen import generate_world

world_gen_jitted = jax.jit(generate_world, static_argnums=(2, ))

for _ in range(100):
    state = world_gen_jitted(rng, env.default_params, env.static_env_params)

import time
times = []
for _ in range(1000):
    start_time = time.time()
    state = world_gen_jitted(rng, env.default_params, env.static_env_params)
    end_time = time.time()
    times.append(end_time - start_time)

mean_time = jnp.mean(jnp.array(times))
median_time = jnp.median(jnp.array(times))

print(f"Mean time per run: {mean_time:.6f} seconds")
print(f"Median time per run: {median_time:.6f} seconds")

# %%
from craftax_marl.game_logic import craftax_step
import time
from jax.lib import xla_bridge

jitted_step = jax.jit(craftax_step, static_argnames=("static_params", ))
