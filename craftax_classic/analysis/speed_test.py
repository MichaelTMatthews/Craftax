import time

import jax
import numpy as np

from craftax_classic.envs.craftax_symbolic_env import CraftaxEnv


def main():
    NUM_ENVS = 1024 * 2
    NUM_STEPS = 64

    env = CraftaxEnv()
    env_params = env.default_params

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rngs = jax.random.split(rng, NUM_ENVS + 1)
    rng, _rngs = rngs[0], rngs[1:]
    _, env_state = jax.vmap(env.reset, in_axes=(0, None))(_rngs, env_params)

    def _vmap_step(rng_and_states, unused):
        rng, states = rng_and_states
        rng, _rng = jax.random.split(rng)
        actions = jax.random.randint(_rng, (NUM_ENVS,), 0, 17)

        rngs = jax.random.split(rng, NUM_ENVS + 1)
        rng, _rngs = rngs[0], rngs[1:]
        obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(_rngs, states, actions, env_params)

        return (rng, env_state), obs

    t0 = time.time()
    jax.lax.scan(jax.jit(_vmap_step), (rng, env_state), (), length=NUM_STEPS)
    t1 = time.time()

    t0 = time.time()
    jax.lax.scan(jax.jit(_vmap_step), (rng, env_state), (), length=NUM_STEPS)
    t1 = time.time()

    print(f"Steps per second: {NUM_STEPS * NUM_ENVS / (t1 - t0)}")


if __name__ == "__main__":
    main()
