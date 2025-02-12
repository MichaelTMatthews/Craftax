# %%
import jax
import jax.numpy as jnp
from craftax_marl.envs.craftax_pixels_env import CraftaxMARLPixelsEnv as CraftaxEnv
from craftax_marl.util.game_logic_utils import *
from craftax_marl.util.maths_utils import *

# %%
rng = jax.random.PRNGKey(0)
env = CraftaxEnv(CraftaxEnv.default_static_params())
obs, state = env.reset(rng, env.default_params)

# %%
state = state.replace(
    player_position=jnp.array([
        [11,11],
        [13,13],
        [15,15],
        [17,17],
    ])
)
block_position = jnp.array([
    [10,10],
    [11,11],
    [13,13],
    [11,11],
])
is_doing_action = jnp.array([True, True, True, True])
env_params = env.default_params
static_params = env.default_static_params()


# %%
in_other_player = (jnp.expand_dims(state.player_position, axis=1) == jnp.expand_dims(block_position, axis=0)).all(axis=2).T
player_interacting_with = jnp.argmax(in_other_player, axis=-1)

is_interacting_with_other_player = jnp.logical_and(
    in_other_player.any(axis=-1),
    is_doing_action,
)
is_player_being_interacted_with = jnp.any(
    jnp.logical_and(
        jnp.arange(static_params.player_count)[:, None] == player_interacting_with,
        is_interacting_with_other_player[None, :]
    ),
    axis=-1
)
is_player_being_revived = jnp.logical_and(
    is_player_being_interacted_with,
    jnp.logical_not(state.player_alive),
)

damage_taken = jnp.zeros(static_params.player_count).at[player_interacting_with].add(
    is_interacting_with_other_player * get_damage_between_players(state, player_interacting_with)
)
damage_taken *= env_params.friendly_fire

new_player_health = jnp.where(
    is_player_being_revived,
    1.0,
    state.player_health - damage_taken,
)


# %%
def interact_player(state, block_position, is_doing_action, env_params, static_params):
    # If other player dead revive, otherwise damage.

    in_other_player = (jnp.expand_dims(state.player_position, axis=1) == jnp.expand_dims(block_position, axis=0)).all(axis=2).T
    player_interacting_with = jnp.argmax(in_other_player, axis=-1)

    is_interacting_with_other_player = jnp.logical_and(
        in_other_player.any(axis=-1),
        is_doing_action,
    )
    is_player_being_interacted_with = jnp.any(
        jnp.logical_and(
            jnp.arange(static_params.player_count)[:, None] == player_interacting_with,
            is_interacting_with_other_player[None, :]
        ),
        axis=-1
    )

    # Revive other players
    is_player_being_revived = jnp.logical_and(
        is_player_being_interacted_with,
        jnp.logical_not(state.player_alive),
    )

    # Damage other players
    damage_taken = jnp.zeros(static_params.player_count).at[player_interacting_with].add(
        is_interacting_with_other_player * get_player_damage_vector(state)
    )

    new_player_health = jnp.where(
        is_player_being_revived,
        1.0,
        state.player_health - damage_taken,
    )
    state = state.replace(
        player_health=new_player_health,
    )
    return state
