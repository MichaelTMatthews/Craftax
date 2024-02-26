import jax.random
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from craftax.constants import BlockType
from craftax.craftax_state import EnvState
from craftax.envs.craftax_symbolic_env import CraftaxEnv


def analyse_state(state: EnvState):
    def _analyse_floor(floor):
        num_coal = (
            (state.map[:, floor] == BlockType.COAL.value).sum(axis=-1).sum(axis=-1)
        )
        num_iron = (
            (state.map[:, floor] == BlockType.IRON.value).sum(axis=-1).sum(axis=-1)
        )
        num_diamond = (
            (state.map[:, floor] == BlockType.DIAMOND.value).sum(axis=-1).sum(axis=-1)
        )
        num_ruby = (
            (state.map[:, floor] == BlockType.RUBY.value).sum(axis=-1).sum(axis=-1)
        )
        num_sapphire = (
            (state.map[:, floor] == BlockType.SAPPHIRE.value).sum(axis=-1).sum(axis=-1)
        )
        num_chest = (
            (state.map[:, floor] == BlockType.CHEST.value).sum(axis=-1).sum(axis=-1)
        )
        print(
            f"Floor {floor} means: coal={num_coal.mean():.1f}, iron={num_iron.mean():.1f}, diamond={num_diamond.mean():.1f},"
            f" ruby={num_ruby.mean():.1f}, sapphire={num_sapphire.mean():.1f}, chest={num_chest.mean():.1f}"
        )
        # print(f"Floor {floor} stds :coal={num_coal.std():.1f}, iron={num_iron.std():.1f}, diamond={num_diamond.std():.1f}, ruby={num_ruby.std():.1f}, sapphire={num_sapphire.std():.1f}")

    for floor in range(9):
        _analyse_floor(floor)


def main():
    num_envs = 1024
    env = CraftaxEnv(CraftaxEnv.default_static_params())
    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rngs = jax.random.split(rng, num_envs)
    obs, env_state = jax.vmap(env.reset)(rngs)

    analyse_state(env_state)


if __name__ == "__main__":
    debug = True
    if debug:
        with jax.disable_jit():
            main()
    else:
        main()
