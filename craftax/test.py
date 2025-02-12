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
env_params = env.default_params
static_params = env.default_static_params()



# %%
def enchant(rng, state: EnvState, action, static_params: StaticEnvParams):
    target_block_position = state.player_position + DIRECTIONS[state.player_direction]
    target_block = state.map[
        state.player_level, target_block_position[:, 0], target_block_position[:, 1]
    ]
    target_block_is_enchantment_table = jnp.logical_or(
        target_block == BlockType.ENCHANTMENT_TABLE_FIRE.value,
        target_block == BlockType.ENCHANTMENT_TABLE_ICE.value,
    )

    enchantment_type = jnp.where(
        target_block == BlockType.ENCHANTMENT_TABLE_FIRE.value, 
        1, 
        2
    )

    num_gems = jnp.where(
        target_block == BlockType.ENCHANTMENT_TABLE_FIRE.value,
        state.inventory.ruby,
        state.inventory.sapphire,
    )

    could_enchant = jnp.logical_and(
        state.player_mana >= 9,
        jnp.logical_and(target_block_is_enchantment_table, num_gems >= 1),
    )
    could_enchant_warrior = jnp.logical_and(
        state.player_specialization == Specialization.WARRIOR.value,
        could_enchant
    )

    is_enchanting_bow = jnp.logical_and(
        could_enchant_warrior,
        jnp.logical_and(action == Action.ENCHANT_BOW.value, state.inventory.bow > 0),
    )

    is_enchanting_sword = jnp.logical_and(
        could_enchant_warrior,
        jnp.logical_and(
            action == Action.ENCHANT_SWORD.value, state.inventory.sword > 0
        ),
    )

    is_enchanting_armour = jnp.logical_and(
        could_enchant,
        jnp.logical_and(
            action == Action.ENCHANT_ARMOUR.value, state.inventory.armour.sum(axis=1) > 0
        ),
    )

    rng, _rng = jax.random.split(rng)
    unenchanted_armour = state.armour_enchantments == 0
    opposite_enchanted_armour = jnp.logical_and(
        state.armour_enchantments != 0, state.armour_enchantments != enchantment_type[:, None]
    )

    armour_targets = (
        unenchanted_armour + (unenchanted_armour.sum(axis=1) == 0)[:, None] * opposite_enchanted_armour
    )

    _rngs = jax.random.split(rng, static_params.player_count+1)
    rng, _rng = _rngs[0], _rngs[1:]
    armour_target = jax.vmap(jax.random.choice, in_axes=(0, None, None, None, 0))(
        _rng, jnp.arange(4), (), True, armour_targets
    )

    is_enchanting = jnp.logical_or(
        is_enchanting_sword, jnp.logical_or(is_enchanting_bow, is_enchanting_armour)
    )

    new_sword_enchantment = (
        is_enchanting_sword * enchantment_type
        + (1 - is_enchanting_sword) * state.sword_enchantment
    )
    new_bow_enchantment = (
        is_enchanting_bow * enchantment_type
        + (1 - is_enchanting_bow) * state.bow_enchantment
    )

    new_armour_enchantments = state.armour_enchantments.at[jnp.arange(static_params.player_count), armour_target].set(
        is_enchanting_armour * enchantment_type
        + (1 - is_enchanting_armour) * state.armour_enchantments[jnp.arange(static_params.player_count), armour_target]
    )

    new_sapphire = state.inventory.sapphire - 1 * is_enchanting * (
        enchantment_type == 2
    )
    new_ruby = state.inventory.ruby - 1 * is_enchanting * (enchantment_type == 1)
    new_mana = state.player_mana - 9 * is_enchanting

    new_achievements = state.achievements.at[:, Achievement.ENCHANT_SWORD.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.ENCHANT_SWORD.value], is_enchanting_sword
        )
    )

    new_achievements = new_achievements.at[:, Achievement.ENCHANT_ARMOUR.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.ENCHANT_ARMOUR.value], is_enchanting_armour
        )
    )

    return state.replace(
        sword_enchantment=new_sword_enchantment,
        bow_enchantment=new_bow_enchantment,
        armour_enchantments=new_armour_enchantments,
        inventory=state.inventory.replace(
            sapphire=new_sapphire,
            ruby=new_ruby,
        ),
        player_mana=new_mana,
        achievements=new_achievements,
    )
# %%
