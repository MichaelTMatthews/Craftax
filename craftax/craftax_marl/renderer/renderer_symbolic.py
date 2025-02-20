import jax
from functools import partial

from craftax_marl.constants import *
from craftax_marl.craftax_state import EnvState, StaticEnvParams
from craftax_marl.util.game_logic_utils import is_boss_vulnerable, get_player_icon_positions


def render_craftax_symbolic(state: EnvState, static_params: StaticEnvParams):
    map = state.map[state.player_level]

    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    # Map
    padded_grid = jnp.pad(
        map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )

    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2

    map_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
        padded_grid, tl_corner, OBS_DIM
    )
    map_view_one_hot = jax.nn.one_hot(map_view, num_classes=len(BlockType))

    # Items
    padded_items_map = jnp.pad(
        state.item_map[state.player_level],
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=ItemType.NONE.value,
    )

    # Create item map view for each player
    item_map_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
        padded_items_map, tl_corner, OBS_DIM
    )
    item_map_view_one_hot = jax.nn.one_hot(item_map_view, num_classes=len(ItemType))

    # Mobs
    mob_types_per_class = 8
    mob_map = jnp.zeros(
        (static_params.player_count, *OBS_DIM, 5 * mob_types_per_class), dtype=jnp.int32
    )  # 5 classes * 8 types

    def _add_mob_to_map(carry, mob_index):
        mob_map, mobs, mob_class_index = carry

        local_position = (
            -1 * state.player_position
            + mobs.position[mob_index]
            + obs_dim_array // 2
        )

        on_screen = jnp.logical_and(
            local_position >= 0, local_position < obs_dim_array
        ).all(axis=-1)
        on_screen *= mobs.mask[mob_index]

        mob_identifier = mob_class_index * mob_types_per_class + mobs.type_id[mob_index]

        def _set_mobs_on_map(mob_map, local_position, on_screen):
            return mob_map.at[local_position[0], local_position[1], mob_identifier].set(
                on_screen.astype(jnp.int32)
            )

        mob_map = jax.vmap(_set_mobs_on_map, in_axes=(0, 0, 0))(
            mob_map, local_position, on_screen
        )

        return (mob_map, mobs, mob_class_index), None

    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (mob_map, jax.tree_map(lambda x: x[state.player_level], state.melee_mobs), 0),
        jnp.arange(state.melee_mobs.mask.shape[1]),
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (mob_map, jax.tree_map(lambda x: x[state.player_level], state.passive_mobs), 1),
        jnp.arange(state.passive_mobs.mask.shape[1]),
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (mob_map, jax.tree_map(lambda x: x[state.player_level], state.ranged_mobs), 2),
        jnp.arange(state.ranged_mobs.mask.shape[1]),
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (
            mob_map,
            jax.tree_map(lambda x: x[state.player_level], state.mob_projectiles),
            3,
        ),
        jnp.arange(state.mob_projectiles.mask.shape[1]),
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (
            mob_map,
            jax.tree_map(lambda x: x[state.player_level], state.player_projectiles),
            4,
        ),
        jnp.arange(state.player_projectiles.mask.shape[1]),
    )
    
    # Teammate map (One-hot encoding of teammate + bit for dead/alive)
    def _add_teammate(player_index):
        """Creates teammate map for each player"""
        teammate_map = jnp.zeros(
            (*OBS_DIM, static_params.player_count + 1), dtype=jnp.int32
        )
        local_position = (
            -1 * state.player_position[player_index]
            + state.player_position
            + obs_dim_array // 2
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < obs_dim_array
        ).all(axis=-1)

        # Add teammate encoding
        teammate_map = teammate_map.at[
            local_position[:, 0], local_position[:, 1], jnp.arange(static_params.player_count)
        ].max(on_screen)

        # Add dead/alive bit
        teammate_map = teammate_map.at[
            local_position[:, 0], local_position[:, 1], -1
        ].set(
            jnp.logical_and(
                on_screen,
                state.player_alive
            )
        )

        """
        Find direction to teammates
        """
        direction_index_2d = jnp.where(
            local_position < 0, 1,
            jnp.where(local_position >= obs_dim_array, 2, 0)
        )
        direction_index = direction_index_2d[:, 0]*3 + direction_index_2d[:, 1] - 1
        teammate_directions = jax.nn.one_hot(direction_index, num_classes=8)

        return teammate_map, teammate_directions
    teammate_map, teammate_directions = jax.vmap(_add_teammate, in_axes=0)(jnp.arange(static_params.player_count))

    # Concat all maps
    all_map = jnp.concatenate(
        [map_view_one_hot, item_map_view_one_hot, mob_map, teammate_map], axis=-1
    )

    # Light map
    padded_light_map = jnp.pad(
        state.light_map[state.player_level],
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=0.0,
    )

    # create light map for each player
    light_map_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
        padded_light_map, tl_corner, OBS_DIM
    )
    light_map_view = light_map_view > 0.05

    # Mask out tiles and mobs in darkness
    all_map = all_map * light_map_view[:, :, :, None]
    all_map = jnp.concatenate(
        (all_map, jnp.expand_dims(light_map_view, axis=-1)), axis=-1
    )

    # Inventory
    inventory = jnp.stack(
        (
            jnp.sqrt(state.inventory.wood) / 10.0,
            jnp.sqrt(state.inventory.stone) / 10.0,
            jnp.sqrt(state.inventory.coal) / 10.0,
            jnp.sqrt(state.inventory.iron) / 10.0,
            jnp.sqrt(state.inventory.diamond) / 10.0,
            jnp.sqrt(state.inventory.sapphire) / 10.0,
            jnp.sqrt(state.inventory.ruby) / 10.0,
            jnp.sqrt(state.inventory.sapling) / 10.0,
            jnp.sqrt(state.inventory.torches) / 10.0,
            jnp.sqrt(state.inventory.arrows) / 10.0,
            state.inventory.books,
            state.inventory.pickaxe / 4.0,
            state.inventory.sword / 4.0,
            state.sword_enchantment,
            state.bow_enchantment,
            state.inventory.bow,
        ),
        axis=1,
        dtype=jnp.float32,
    )

    potions = jnp.sqrt(state.inventory.potions) / 10.0
    armour = state.inventory.armour / 2.0
    armour_enchantments = state.armour_enchantments

    intrinsics = jnp.stack(
        (
            # state.player_health / 10.0, -- Removed and placed as part of the teammate dashboard
            state.player_food / 10.0,
            state.player_drink / 10.0,
            state.player_energy / 10.0,
            state.player_mana / 10.0,
            state.player_xp / 10.0,
            state.player_dexterity / 10.0,
            state.player_strength / 10.0,
            state.player_intelligence / 10.0,
        ),
        axis=1,
        dtype=jnp.float32,
    )

    direction = jax.nn.one_hot(state.player_direction - 1, num_classes=4)

    special_values_per_player = jnp.stack(
        (
            state.is_sleeping,
            state.is_resting,
            state.learned_spells,
        ),
        axis=1,
    )
    special_values_level = jnp.array(
        [
            state.light_level,
            state.player_level / 10.0,
            state.monsters_killed[state.player_level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL,
            is_boss_vulnerable(state),
        ]
    )

    """
    Teammate Dashboard
        Includes:
            - Player Health
            - Player Dead or Alive
            - Specialization
            - Requested Material
    Teammate Dashboard appears the same for all players
    """
    players_health = state.player_health / 10.0
    players_alive = state.player_alive
    players_specialization = jax.nn.one_hot(state.player_specialization - Specialization.FORAGER.value, num_classes=3)
    requested_material = (
        jax.nn.one_hot(
            state.request_type - Action.REQUEST_FOOD.value, 
            num_classes=(Action.REQUEST_SAPPHIRE.value - Action.REQUEST_FOOD.value + 1)
        )
        * (state.request_duration > 0)[:, None]
    )
    teammate_dashboard = jnp.concatenate(
        (players_health[:, None], players_alive[:, None], players_specialization, requested_material),
        axis=-1
    ).flatten()
    teammate_dashboard = jnp.repeat(teammate_dashboard[None, ...], 3, axis=0)

    all_flattened = jnp.concatenate(
        [
            all_map.reshape(all_map.shape[0], -1),
            teammate_dashboard,
            teammate_directions.reshape(teammate_directions.shape[0], -1),
            inventory,
            potions,
            intrinsics,
            direction,
            armour,
            armour_enchantments,
            special_values_per_player,
            special_values_level[None, :].repeat(static_params.player_count, axis=0),
        ],
        axis=1,
    )

    return all_flattened
