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

    # Chest
    def _create_chest_map(player_index):
        local_position = (
            state.chest_positions[static_params.player_level]
            - state.player_position[player_index, None]
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        in_bounds = jnp.logical_and(local_position >= 0, local_position < obs_dim_array).all(axis=-1)
        is_still_chest = state.map[
            static_params.player_level, 
            state.chest_positions[static_params.player_level, :, :, 0], 
            state.chest_positions[static_params.player_level, :, :, 1]
        ] == BlockType.CHEST.value
        placed_chest = jnp.logical_and(in_bounds, is_still_chest)
        row_indices = jnp.arange(placed_chest.shape[0]).reshape(-1, 1)
        row_indices_array = jnp.tile(row_indices, (1, placed_chest.shape[1]))

        player_chest_map = jnp.zeros((static_params.player_count, *OBS_DIM))
        player_chest_map = player_chest_map.at[
            row_indices_array, 
            local_position[..., 0], 
            local_position[..., 1]
        ].max(placed_chest)
        return player_chest_map

    chest_map_view = jax.vmap(_create_chest_map)(
        jnp.arange(static_params.player_count)
    )

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
            + jnp.array([OBS_DIM[0], OBS_DIM[1]]) // 2
        )

        on_screen = jnp.logical_and(
            local_position >= 0, local_position < jnp.array([OBS_DIM[0], OBS_DIM[1]])
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
    
    # Teammate map (0th index for alive, 1th for dead)
    def _add_teammate(player_index):
        """Creates teammate map for each player"""
        teammate_map = jnp.zeros(
            (static_params.player_count, *OBS_DIM, 2), dtype=jnp.int32
        )
        local_position = (
            -1 * state.player_position[player_index]
            + state.player_position
            + jnp.array([OBS_DIM[0], OBS_DIM[1]]) // 2
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < jnp.array([OBS_DIM[0], OBS_DIM[1]])
        ).all(axis=-1)
        teammate_map = teammate_map.at[
            jnp.arange(static_params.player_count), local_position[:, 0], local_position[:, 1], 0
        ].set(
            jnp.logical_and(
                on_screen,
                state.player_alive
            )
        )
        teammate_map = teammate_map.at[
            jnp.arange(static_params.player_count), local_position[:, 0], local_position[:, 1], 1
        ].set(
            jnp.logical_and(
                on_screen,
                jnp.logical_not(state.player_alive)
            )
        )
        return teammate_map
    teammate_map = jax.vmap(_add_teammate, in_axes=0)(jnp.arange(static_params.player_count))

    # Teammate direction for off-screen teammates (0th index for alive, 1th for dead)
    teammate_direction = jnp.zeros(
        (static_params.player_count, 8, 2)
    )
    local_position = (
        -1 * state.player_position[:, None]
        + state.player_position
        + jnp.array([OBS_DIM[0], OBS_DIM[1]]) // 2
    )
    on_screen = jnp.logical_and(
        local_position >= 0, local_position < jnp.array([OBS_DIM[0], OBS_DIM[1]])
    ).all(axis=-1)
    direction_index_2d = jnp.where(
        local_position < 0, 1,
        jnp.where(local_position >= jnp.array([OBS_DIM[0], OBS_DIM[1]]), 2, 0)
    )
    direction_index = direction_index_2d[:, :, 0]*3 + direction_index_2d[:, :, 1] - 1
    teammate_direction = teammate_direction.at[
        jnp.arange(static_params.player_count)[:, None], direction_index, 0
    ].max(
        jnp.logical_and(
            jnp.logical_not(on_screen),
            state.player_alive
        )
    )
    teammate_direction = teammate_direction.at[
        jnp.arange(static_params.player_count)[:, None], direction_index, 1
    ].max(
        jnp.logical_and(
            jnp.logical_not(on_screen),
            jnp.logical_not(state.player_alive)
        )
    )

    # Concat all maps
    all_map = jnp.concatenate(
        [map_view_one_hot, item_map_view_one_hot, mob_map, teammate_map, chest_map_view], axis=-1
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
            state.inventory.books / 2.0,
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
            state.player_health / 10.0,
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
            state.learned_spells[:, 0],
            state.learned_spells[:, 1],
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

    all_flattened = jnp.concatenate(
        [
            all_map.reshape(all_map.shape[0], -1),
            teammate_direction.reshape(teammate_direction.shape[0], -1),
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


@partial(
    jax.jit,
    static_argnums=(
        1,
        2,
    ),
)
def render_craftax_pixels(state, block_pixel_size, static_params, player_specific_textures, do_night_noise=True):
    textures = TEXTURES[block_pixel_size]
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    # RENDER MAP
    # Get view of map
    map = state.map[state.player_level]
    padded_grid = jnp.pad(
        map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )

    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2

    map_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
        padded_grid, tl_corner, OBS_DIM
    )

    # Boss
    boss_block = jax.lax.select(
        is_boss_vulnerable(state),
        BlockType.NECROMANCER_VULNERABLE.value,
        BlockType.NECROMANCER.value,
    )

    map_view_boss = map_view == BlockType.NECROMANCER.value
    map_view = map_view_boss * boss_block + (1 - map_view_boss) * map_view

    # Render map tiles
    map_pixels_indexes = jnp.repeat(
        jnp.repeat(map_view, repeats=block_pixel_size, axis=1),
        repeats=block_pixel_size,
        axis=2,
    )
    map_pixels_indexes = jnp.expand_dims(map_pixels_indexes, axis=-1)
    map_pixels_indexes = jnp.repeat(map_pixels_indexes, repeats=3, axis=-1)

    map_pixels = jnp.zeros_like(map_pixels_indexes, dtype=jnp.float32)

    def _add_block_type_to_pixels(pixels, block_index):
        return (
            pixels
            + textures["full_map_block_textures"][block_index]
            * (map_pixels_indexes == block_index),
            None,
        )

    map_pixels, _ = jax.lax.scan(
        _add_block_type_to_pixels, map_pixels, jnp.arange(len(BlockType))
    )

    # Render Colored Chests
    def _add_player_chests(chest_map_view, player_index):
        """Adds players chest position to other players"""
        local_position = (
            state.chest_positions[state.player_level, player_index]
            - state.player_position[:, None]
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        def _single_batch_index(data_slice, row_idx, col_idx):
            return data_slice.at[row_idx, col_idx].set(player_index)
        chest_map_view = jax.vmap(_single_batch_index, in_axes=(0,0,0))(
            chest_map_view, local_position[..., 0], local_position[..., 1]
        )
        return chest_map_view, None

    chest_map_view = jnp.full_like(map_view, fill_value=-1)
    chest_map_view, _ = jax.lax.scan(
        _add_player_chests,
        chest_map_view,
        jnp.arange(static_params.player_count),
    )
    chest_map_pixels_indexes = jnp.repeat(
        jnp.repeat(chest_map_view, repeats=block_pixel_size, axis=1),
        repeats=block_pixel_size,
        axis=2,
    )
    chest_map_pixels_indexes = jnp.expand_dims(chest_map_pixels_indexes, axis=-1)
    chest_map_pixels_indexes = jnp.repeat(chest_map_pixels_indexes, repeats=3, axis=-1)

    def _add_player_chest_to_pixels(pixels, player_index):
        return (
            pixels
            + (player_specific_textures.chest_textures[player_index] - pixels)
            * (chest_map_pixels_indexes == player_index)
            * (map_pixels_indexes == BlockType.CHEST.value),
            None,
        )

    map_pixels, _ = jax.lax.scan(
        _add_player_chest_to_pixels, map_pixels, jnp.arange(static_params.player_count)
    )

    # Items
    padded_item_map = jnp.pad(
        state.item_map[state.player_level],
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=ItemType.NONE.value,
    )

    item_map_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
        padded_item_map, tl_corner, OBS_DIM
    )

    # Insert blocked ladders
    is_ladder_down_open = (
        state.monsters_killed[state.player_level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL
    )
    ladder_down_item = jax.lax.select(
        is_ladder_down_open,
        ItemType.LADDER_DOWN.value,
        ItemType.LADDER_DOWN_BLOCKED.value,
    )

    item_map_view_is_ladder_down = item_map_view == ItemType.LADDER_DOWN.value
    item_map_view = (
        item_map_view_is_ladder_down * ladder_down_item
        + (1 - item_map_view_is_ladder_down) * item_map_view
    )

    map_pixels_item_indexes = jnp.repeat(
        jnp.repeat(item_map_view, repeats=block_pixel_size, axis=1),
        repeats=block_pixel_size,
        axis=2,
    )
    map_pixels_item_indexes = jnp.expand_dims(map_pixels_item_indexes, axis=-1)
    map_pixels_item_indexes = jnp.repeat(map_pixels_item_indexes, repeats=3, axis=-1)

    def _add_item_type_to_pixels(pixels, item_index):
        full_map_texture = textures["full_map_item_textures"][item_index]
        mask = map_pixels_item_indexes == item_index

        pixels = (
            pixels * (1 - full_map_texture[:, :, 3] * mask[:, :, :, 0])[:, :, :, None]
        )
        pixels = (
            pixels
            + full_map_texture[:, :, :3] * mask * full_map_texture[:, :, 3][:, :, None]
        )

        return pixels, None

    map_pixels, _ = jax.lax.scan(
        _add_item_type_to_pixels, map_pixels, jnp.arange(1, len(ItemType))
    )

    # Render player
    # Helper functions to display and update slice
    def _slice_pixel_map(player_pixels, local_position):
        return jax.lax.dynamic_slice(
            player_pixels,
            (
                local_position[0] * block_pixel_size,
                local_position[1] * block_pixel_size,
                0,
            ),
            (block_pixel_size, block_pixel_size, 3),
        )

    def _update_slice_pixel_map(player_pixels, texture_with_background, local_position):
        return jax.lax.dynamic_update_slice(
            player_pixels,
            texture_with_background,
            (
                local_position[0] * block_pixel_size,
                local_position[1] * block_pixel_size,
                0,
            ),
        )

    # Render each player on the map of other players
    def _render_friends(pixels, player_index):
        local_position = (
            state.player_position[player_index]
            - state.player_position
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < obs_dim_array
        ).all(axis=-1)

        player_texture_index = jax.lax.select(
            state.is_sleeping[player_index], 4, state.player_direction[player_index] - 1
        )
        player_texture_index = jax.lax.select(
            state.player_alive[player_index], player_texture_index, 5
        )
        player_texture = player_specific_textures.player_textures[player_index, player_texture_index]
        player_texture, player_texture_alpha = (
            player_texture[:, :, :3],
            player_texture[:, :, 3:],
        )

        player_texture = jax.vmap(jnp.multiply, in_axes=(None, 0))(
            player_texture, on_screen
        )
        player_texture_with_background = 1 - jax.vmap(jnp.multiply, in_axes=(None, 0))(
            player_texture_alpha, on_screen
        )
        player_texture_with_background = player_texture_with_background * jax.vmap(
            _slice_pixel_map, in_axes=(0, 0)
        )(pixels, local_position)
        player_texture_with_background = (
            player_texture_with_background + player_texture * player_texture_alpha
        )

        pixels = jax.vmap(_update_slice_pixel_map, in_axes=(0, 0, 0))(
            pixels, player_texture_with_background, local_position
        )
        return pixels, None

    map_pixels, _ = jax.lax.scan(
        _render_friends, map_pixels, jnp.arange(static_params.player_count)
    )

    # Render mobs
    # Zombies
    def _add_mob_to_pixels(carry, mob_index):
        pixels, mobs, texture_name, alpha_texture_name = carry
        local_position = (
            mobs.position[state.player_level, mob_index]
            - state.player_position
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < obs_dim_array
        ).all(axis=-1)
        on_screen *= mobs.mask[state.player_level, mob_index]

        mob_texture = texture_name[mobs.type_id[state.player_level, mob_index]]
        mob_texture_alpha = alpha_texture_name[
            mobs.type_id[state.player_level, mob_index]
        ]

        mob_texture = jax.vmap(jnp.multiply, in_axes=(None, 0))(mob_texture, on_screen)

        mob_texture_with_background = 1 - jax.vmap(jnp.multiply, in_axes=(None, 0))(
            mob_texture_alpha, on_screen
        )

        mob_texture_with_background = mob_texture_with_background * jax.vmap(
            _slice_pixel_map, in_axes=(0, 0)
        )(pixels, local_position)

        mob_texture_with_background = (
            mob_texture_with_background + mob_texture * mob_texture_alpha
        )

        pixels = jax.vmap(_update_slice_pixel_map, in_axes=(0, 0, 0))(
            pixels, mob_texture_with_background, local_position
        )

        return (pixels, mobs, texture_name, alpha_texture_name), None

    (map_pixels, _, _, _), _ = jax.lax.scan(
        _add_mob_to_pixels,
        (
            map_pixels,
            state.melee_mobs,
            textures["melee_mob_textures"],
            textures["melee_mob_texture_alphas"],
        ),
        jnp.arange(state.melee_mobs.mask.shape[1]),
    )

    (map_pixels, _, _, _), _ = jax.lax.scan(
        _add_mob_to_pixels,
        (
            map_pixels,
            state.passive_mobs,
            textures["passive_mob_textures"],
            textures["passive_mob_texture_alphas"],
        ),
        jnp.arange(state.passive_mobs.mask.shape[1]),
    )

    (map_pixels, _, _, _), _ = jax.lax.scan(
        _add_mob_to_pixels,
        (
            map_pixels,
            state.ranged_mobs,
            textures["ranged_mob_textures"],
            textures["ranged_mob_texture_alphas"],
        ),
        jnp.arange(state.ranged_mobs.mask.shape[1]),
    )

    def _add_projectile_to_pixels(carry, projectile_index):
        pixels, projectiles, projectile_directions = carry
        local_position = (
            projectiles.position[state.player_level, projectile_index]
            - state.player_position
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < obs_dim_array
        ).all(axis=-1)
        on_screen *= projectiles.mask[state.player_level, projectile_index]

        projectile_texture = textures["projectile_textures"][
            projectiles.type_id[state.player_level, projectile_index]
        ]
        projectile_texture_alpha = textures["projectile_texture_alphas"][
            projectiles.type_id[state.player_level, projectile_index]
        ]

        flipped_projectile_texture = jnp.flip(projectile_texture, axis=0)
        flipped_projectile_texture_alpha = jnp.flip(projectile_texture_alpha, axis=0)
        flip_projectile = jnp.logical_or(
            projectile_directions[state.player_level, projectile_index, 0] > 0,
            projectile_directions[state.player_level, projectile_index, 1] > 0,
        )

        projectile_texture = jax.lax.select(
            flip_projectile,
            flipped_projectile_texture,
            projectile_texture,
        )
        projectile_texture_alpha = jax.lax.select(
            flip_projectile,
            flipped_projectile_texture_alpha,
            projectile_texture_alpha,
        )

        transposed_projectile_texture = jnp.transpose(projectile_texture, (1, 0, 2))
        transposed_projectile_texture_alpha = jnp.transpose(
            projectile_texture_alpha, (1, 0, 2)
        )

        projectile_texture = jax.lax.select(
            projectile_directions[state.player_level, projectile_index, 1] != 0,
            transposed_projectile_texture,
            projectile_texture,
        )
        projectile_texture_alpha = jax.lax.select(
            projectile_directions[state.player_level, projectile_index, 1] != 0,
            transposed_projectile_texture_alpha,
            projectile_texture_alpha,
        )

        projectile_texture = jax.vmap(jnp.multiply, in_axes=(None, 0))(
            projectile_texture, on_screen
        )
        projectile_texture_with_background = 1 - jax.vmap(
            jnp.multiply, in_axes=(None, 0)
        )(projectile_texture_alpha, on_screen)

        projectile_texture_with_background = (
            projectile_texture_with_background
            * jax.vmap(_slice_pixel_map, in_axes=(0, 0))(pixels, local_position)
        )

        projectile_texture_with_background = (
            projectile_texture_with_background
            + projectile_texture * projectile_texture_alpha
        )

        pixels = jax.vmap(_update_slice_pixel_map, in_axes=(0, 0, 0))(
            pixels, projectile_texture_with_background, local_position
        )

        return (pixels, projectiles, projectile_directions), None

    (map_pixels, _, _), _ = jax.lax.scan(
        _add_projectile_to_pixels,
        (map_pixels, state.mob_projectiles, state.mob_projectile_directions),
        jnp.arange(state.mob_projectiles.mask.shape[1]),
    )

    (map_pixels, _, _), _ = jax.lax.scan(
        _add_projectile_to_pixels,
        (map_pixels, state.player_projectiles, state.player_projectile_directions),
        jnp.arange(state.player_projectiles.mask.shape[1]),
    )

    # Apply darkness (underground)
    light_map = state.light_map[state.player_level]
    padded_light_map = jnp.pad(
        light_map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=False,
    )

    light_map_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
        padded_light_map, tl_corner, OBS_DIM
    )
    light_map_pixels = light_map_view.repeat(block_pixel_size, axis=1).repeat(
        block_pixel_size, axis=2
    )

    map_pixels = (light_map_pixels)[:, :, :, None] * map_pixels

    # Apply night
    night_pixels = textures["night_texture"]
    daylight = state.light_level
    daylight = jax.lax.select(state.player_level == 0, daylight, 1.0)

    if do_night_noise:
        night_noise = (
            jax.random.uniform(state.state_rng, night_pixels.shape[:2]) * 95 + 32
        )
        night_noise = jnp.expand_dims(night_noise, axis=-1).repeat(3, axis=-1)

        night_intensity = 2 * (0.5 - daylight)
        night_intensity = jnp.maximum(night_intensity, 0.0)
        night_mask = textures["night_noise_intensity_texture"] * night_intensity
        night = (1.0 - night_mask) * map_pixels + night_mask * night_noise

        night = night_pixels * 0.5 + 0.5 * night
        map_pixels = daylight * map_pixels + (1 - daylight) * night
    else:
        night_noise = jnp.full(night_pixels.shape, 64)

        night_intensity = 2 * (0.5 - daylight)
        night_intensity = jnp.maximum(night_intensity, 0.0)
        night_mask = (
            jnp.ones_like(textures["night_noise_intensity_texture"])
            * night_intensity
            * 0.5
        )
        night = (1.0 - night_mask) * map_pixels + night_mask * night_noise

        night = night_pixels * 0.5 + 0.5 * night
        map_pixels = daylight * map_pixels + (1 - daylight) * night
        # map_pixels = daylight * map_pixels
        # night_noise = jnp.ones(night_pixels.shape[:2]) * 64

    # Apply sleep
    sleep_level = 1.0 - state.is_sleeping * 0.5
    map_pixels = jax.vmap(jnp.multiply, in_axes=(0, 0))(sleep_level, map_pixels)

    # RENDER INVENTORY
    inv_pixel_left_space = (block_pixel_size - int(0.8 * block_pixel_size)) // 2
    inv_pixel_right_space = (
        block_pixel_size - int(0.8 * block_pixel_size) - inv_pixel_left_space
    )

    inv_pixels = jnp.zeros(
        (
            map_pixels.shape[0],
            INVENTORY_OBS_HEIGHT * block_pixel_size,
            OBS_DIM[1] * block_pixel_size,
            3,
        ),
        dtype=jnp.float32,
    )

    number_size = int(block_pixel_size * 0.4)
    number_offset = block_pixel_size - number_size
    number_double_offset = block_pixel_size - 2 * number_size

    def _render_digit(pixels, number, x, y):
        pixels = pixels.at[
            y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
            x * block_pixel_size + number_offset : (x + 1) * block_pixel_size,
        ].mul(1 - textures["number_textures_alpha"][number])

        pixels = pixels.at[
            y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
            x * block_pixel_size + number_offset : (x + 1) * block_pixel_size,
        ].add(textures["number_textures"][number])

        return pixels

    def _render_two_digit_number(pixels, number, x, y):
        tens = number // 10
        ones = number % 10

        ones_textures = jax.lax.select(
            number == 0,
            textures["number_textures"],
            textures["number_textures_with_zero"],
        )

        ones_textures_alpha = jax.lax.select(
            number == 0,
            textures["number_textures_alpha"],
            textures["number_textures_alpha_with_zero"],
        )

        pixels = pixels.at[
            y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
            x * block_pixel_size + number_offset : (x + 1) * block_pixel_size,
        ].mul(1 - ones_textures_alpha[ones])

        pixels = pixels.at[
            y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
            x * block_pixel_size + number_offset : (x + 1) * block_pixel_size,
        ].add(ones_textures[ones])

        pixels = pixels.at[
            y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
            x * block_pixel_size
            + number_double_offset : x * block_pixel_size
            + number_offset,
        ].mul(1 - textures["number_textures_alpha"][tens])

        pixels = pixels.at[
            y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
            x * block_pixel_size
            + number_double_offset : x * block_pixel_size
            + number_offset,
        ].add(textures["number_textures"][tens])

        return pixels

    def _render_icon(pixels, texture, x, y):
        return pixels.at[
            block_pixel_size * y
            + inv_pixel_left_space : block_pixel_size * (y + 1)
            - inv_pixel_right_space,
            block_pixel_size * x
            + inv_pixel_left_space : block_pixel_size * (x + 1)
            - inv_pixel_right_space,
        ].set(texture)

    def _render_icon_with_alpha(pixels, texture, x, y):
        existing_slice = pixels[
            block_pixel_size * y
            + inv_pixel_left_space : block_pixel_size * (y + 1)
            - inv_pixel_right_space,
            block_pixel_size * x
            + inv_pixel_left_space : block_pixel_size * (x + 1)
            - inv_pixel_right_space,
        ]

        new_slice = (
            existing_slice * (1 - texture[:, :, 3][:, :, None])
            + texture[:, :, :3] * texture[:, :, 3][:, :, None]
        )

        return pixels.at[
            block_pixel_size * y
            + inv_pixel_left_space : block_pixel_size * (y + 1)
            - inv_pixel_right_space,
            block_pixel_size * x
            + inv_pixel_left_space : block_pixel_size * (x + 1)
            - inv_pixel_right_space,
        ].set(new_slice)

    def _render_icons(pixels, textures, locs):
        def _render_single_icon(carry, idx):
            pixels, textures, locs = carry
            icon_slice = textures[idx]
            pixels = jax.lax.dynamic_update_slice(
                pixels,
                icon_slice,
                (
                    block_pixel_size * locs[idx, 0] + inv_pixel_left_space,
                    block_pixel_size * locs[idx, 1] + inv_pixel_left_space,
                    0,
                ),
            )
            return (pixels, textures, locs), None
        (pixels, _, _), _ = jax.lax.scan(
            _render_single_icon, (pixels, textures, locs), jnp.arange(locs.shape[0])
        )
        return pixels
    
    def _render_two_digit_numbers(pixels, numbers, locs):
        tens = numbers // 10
        ones = numbers % 10

        ones_textures = jnp.where(
            (numbers == 0)[:, None, None, None, None],
            textures["number_textures"],
            textures["number_textures_with_zero"],
        )

        ones_textures_alpha = jnp.where(
            (numbers == 0)[:, None, None, None, None],
            textures["number_textures_alpha"],
            textures["number_textures_alpha_with_zero"],
        )

        def _render_single_two_digit_number(pixels, idx):
            ones_texture = ones_textures[idx, ones[idx]]
            ones_texture_alpha = ones_textures_alpha[idx, ones[idx]]
            tens_texture = textures["number_textures"][tens[idx]]
            tens_texture_alpha = textures["number_textures_alpha"][tens[idx]]
            
            # Render Ones
            original_ones_slice = jax.lax.dynamic_slice(
                pixels,
                (
                    block_pixel_size * locs[idx, 0] + number_offset,
                    block_pixel_size * locs[idx, 1] + number_offset,
                    0,
                ),
                (number_size, number_size, 3),
            )
            updated_ones_slice = (
                original_ones_slice 
                * (1 - ones_texture_alpha)
                + ones_texture
            )
            pixels = jax.lax.dynamic_update_slice(
                pixels,
                updated_ones_slice,
                (
                    block_pixel_size * locs[idx, 0] + number_offset,
                    block_pixel_size * locs[idx, 1] + number_offset,
                    0,
                ),
            )

            # Render Tens
            original_tens_slice = jax.lax.dynamic_slice(
                pixels,
                (
                    block_pixel_size * locs[idx, 0] + number_offset,
                    block_pixel_size * locs[idx, 1] + number_double_offset,
                    0,
                ),
                (number_size, number_size, 3),
            )
            updated_tens_slice = (
                original_tens_slice 
                * (1 - tens_texture_alpha)
                + tens_texture
            )
            pixels = jax.lax.dynamic_update_slice(
                pixels,
                updated_tens_slice,
                (
                    block_pixel_size * locs[idx, 0] + number_offset,
                    block_pixel_size * locs[idx, 1] + number_double_offset,
                    0,
                ),
            )

            return pixels, None
        
        pixels, _ = jax.lax.scan(
            _render_single_two_digit_number, pixels, jnp.arange(static_params.player_count)
        )

        return pixels

    def _render_dashboard(inv_pixels, player_index):
        # Render player stats
        player_health = jnp.maximum(
            jnp.floor(state.player_health[player_index]), 1
        ).astype(int)
        health_texture = jax.lax.select(
            player_health > 0,
            textures["health_texture"],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, health_texture, 0, 0)
        inv_pixels = _render_two_digit_number(inv_pixels, player_health, 0, 0)

        hunger_texture = jax.lax.select(
            state.player_food[player_index] > 0,
            textures["hunger_texture"],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, hunger_texture, 1, 0)
        inv_pixels = _render_two_digit_number(
            inv_pixels, state.player_food[player_index], 1, 0
        )

        thirst_texture = jax.lax.select(
            state.player_drink[player_index] > 0,
            textures["thirst_texture"],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, thirst_texture, 2, 0)
        inv_pixels = _render_two_digit_number(
            inv_pixels, state.player_drink[player_index], 2, 0
        )

        energy_texture = jax.lax.select(
            state.player_energy[player_index] > 0,
            textures["energy_texture"],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, energy_texture, 3, 0)
        inv_pixels = _render_two_digit_number(
            inv_pixels, state.player_energy[player_index], 3, 0
        )

        mana_texture = jax.lax.select(
            state.player_mana[player_index] > 0,
            textures["mana_texture"],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, mana_texture, 4, 0)
        inv_pixels = _render_two_digit_number(
            inv_pixels, state.player_mana[player_index], 4, 0
        )

        # Render inventory

        inv_wood_texture = jax.lax.select(
            state.inventory.wood[player_index] > 0,
            textures["smaller_block_textures"][BlockType.WOOD.value],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, inv_wood_texture, 0, 2)
        inv_pixels = _render_two_digit_number(
            inv_pixels, state.inventory.wood[player_index], 0, 2
        )

        inv_stone_texture = jax.lax.select(
            state.inventory.stone[player_index] > 0,
            textures["smaller_block_textures"][BlockType.STONE.value],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, inv_stone_texture, 1, 2)
        inv_pixels = _render_two_digit_number(
            inv_pixels, state.inventory.stone[player_index], 1, 2
        )

        inv_coal_texture = jax.lax.select(
            state.inventory.coal[player_index] > 0,
            textures["smaller_block_textures"][BlockType.COAL.value],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, inv_coal_texture, 0, 1)
        inv_pixels = _render_two_digit_number(
            inv_pixels, state.inventory.coal[player_index], 0, 1
        )

        inv_iron_texture = jax.lax.select(
            state.inventory.iron[player_index] > 0,
            textures["smaller_block_textures"][BlockType.IRON.value],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, inv_iron_texture, 1, 1)
        inv_pixels = _render_two_digit_number(
            inv_pixels, state.inventory.iron[player_index], 1, 1
        )

        inv_diamond_texture = jax.lax.select(
            state.inventory.diamond[player_index] > 0,
            textures["smaller_block_textures"][BlockType.DIAMOND.value],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, inv_diamond_texture, 2, 1)
        inv_pixels = _render_two_digit_number(
            inv_pixels, state.inventory.diamond[player_index], 2, 1
        )

        inv_sapphire_texture = jax.lax.select(
            state.inventory.sapphire[player_index] > 0,
            textures["smaller_block_textures"][BlockType.SAPPHIRE.value],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, inv_sapphire_texture, 3, 1)
        inv_pixels = _render_two_digit_number(
            inv_pixels, state.inventory.sapphire[player_index], 3, 1
        )

        inv_ruby_texture = jax.lax.select(
            state.inventory.ruby[player_index] > 0,
            textures["smaller_block_textures"][BlockType.RUBY.value],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, inv_ruby_texture, 4, 1)
        inv_pixels = _render_two_digit_number(
            inv_pixels, state.inventory.ruby[player_index], 4, 1
        )

        inv_sapling_texture = jax.lax.select(
            state.inventory.sapling[player_index] > 0,
            textures["sapling_texture"],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, inv_sapling_texture, 5, 1)
        inv_pixels = _render_two_digit_number(
            inv_pixels, state.inventory.sapling[player_index], 5, 1
        )

        # Render tools
        # Pickaxe
        pickaxe_texture = textures["pickaxe_textures"][
            state.inventory.pickaxe[player_index]
        ]
        inv_pixels = _render_icon(inv_pixels, pickaxe_texture, 8, 2)

        # Sword
        sword_texture = textures["sword_textures"][state.inventory.sword[player_index]]
        inv_pixels = _render_icon(inv_pixels, sword_texture, 8, 1)

        # Bow and arrows
        bow_texture = textures["bow_textures"][state.inventory.bow[player_index]]
        inv_pixels = _render_icon(inv_pixels, bow_texture, 6, 1)

        arrow_texture = jax.lax.select(
            state.inventory.arrows[player_index] > 0,
            textures["player_projectile_textures"][0],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, arrow_texture, 6, 2)
        inv_pixels = _render_two_digit_number(
            inv_pixels, state.inventory.arrows[player_index], 6, 2
        )

        # Armour
        for i in range(4):
            armour_texture = textures["armour_textures"][
                state.inventory.armour[player_index][i], i
            ]
            inv_pixels = _render_icon(inv_pixels, armour_texture, 7, i)

        # Torch
        torch_texture = jax.lax.select(
            state.inventory.torches[player_index] > 0,
            textures["torch_inv_texture"],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, torch_texture, 2, 2)
        inv_pixels = _render_two_digit_number(
            inv_pixels, state.inventory.torches[player_index], 2, 2
        )

        # Potions
        potion_names = ["red", "green", "blue", "pink", "cyan", "yellow"]
        for potion_index, potion_name in enumerate(potion_names):
            potion_texture = jax.lax.select(
                state.inventory.potions[player_index][potion_index] > 0,
                textures["potion_textures"][potion_index],
                textures["smaller_empty_texture"],
            )
            inv_pixels = _render_icon(inv_pixels, potion_texture, potion_index, 3)
            inv_pixels = _render_two_digit_number(
                inv_pixels,
                state.inventory.potions[player_index][potion_index],
                potion_index,
                3,
            )

        # Books
        book_texture = jax.lax.select(
            state.inventory.books[player_index] > 0,
            textures["book_texture"],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, book_texture, 3, 2)
        inv_pixels = _render_two_digit_number(
            inv_pixels, state.inventory.books[player_index], 3, 2
        )

        # Learned spells
        fireball_texture = jax.lax.select(
            state.learned_spells[player_index][0],
            textures["fireball_inv_texture"],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, fireball_texture, 4, 2)

        iceball_texture = jax.lax.select(
            state.learned_spells[player_index][1],
            textures["iceball_inv_texture"],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, iceball_texture, 5, 2)

        # Enchantments
        sword_enchantment_texture = textures["sword_enchantment_textures"][
            state.sword_enchantment[player_index]
        ]
        inv_pixels = _render_icon_with_alpha(
            inv_pixels, sword_enchantment_texture, 8, 1
        )

        arrow_enchantment_level = state.bow_enchantment[player_index] * (
            state.inventory.arrows[player_index] > 0
        )
        arrow_enchantment_texture = textures["arrow_enchantment_textures"][
            arrow_enchantment_level
        ]
        inv_pixels = _render_icon_with_alpha(
            inv_pixels, arrow_enchantment_texture, 6, 2
        )

        for i in range(4):
            armour_enchantment_texture = textures["armour_enchantment_textures"][
                state.armour_enchantments[player_index][i], i
            ]
            inv_pixels = _render_icon_with_alpha(
                inv_pixels, armour_enchantment_texture, 7, i
            )

        # Dungeon level
        inv_pixels = _render_digit(inv_pixels, state.player_level, 6, 0)

        # Attributes
        xp_texture = jax.lax.select(
            state.player_xp[player_index] > 0,
            textures["xp_texture"],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, xp_texture, 9, 0)
        inv_pixels = _render_digit(inv_pixels, state.player_xp[player_index], 9, 0)

        inv_pixels = _render_icon(inv_pixels, textures["dex_texture"], 9, 1)
        inv_pixels = _render_digit(
            inv_pixels, state.player_dexterity[player_index], 9, 1
        )

        inv_pixels = _render_icon(inv_pixels, textures["str_texture"], 9, 2)
        inv_pixels = _render_digit(
            inv_pixels, state.player_strength[player_index], 9, 2
        )

        inv_pixels = _render_icon(inv_pixels, textures["int_texture"], 9, 3)
        inv_pixels = _render_digit(
            inv_pixels, state.player_intelligence[player_index], 9, 3
        )

        return inv_pixels

    inv_pixels = jax.vmap(_render_dashboard, in_axes=(0, 0))(
        inv_pixels, jnp.arange(static_params.player_count)
    )

    def _render_teammate_info(player_index):
        info_pixels = jnp.zeros(
            (
                (static_params.player_count+1)//2 * block_pixel_size,
                OBS_DIM[1] * block_pixel_size,
                3,
            ),
            dtype=jnp.float32,
        )

        # quick icon location calc
        # player_icon_locations = get_player_icon_positions(static_params.player_count) 
        player_icon_locations = get_player_icon_positions(static_params.player_count)

        # Render players icons
        player_icon_to_render = jnp.where(
            state.player_alive[:, None, None, None],
            player_specific_textures.player_icon_textures[:, 0],
            player_specific_textures.player_icon_textures[:, 1],
        )

        info_pixels = _render_icons(info_pixels, player_icon_to_render, player_icon_locations)
        
        # Render teammate healths
        health_icon_locations = player_icon_locations + jnp.array([0, 1])
        teammate_health = jnp.maximum(
            jnp.floor(state.player_health), 1
        ).astype(int)
        health_texture = jnp.where(
            (teammate_health > 0)[:, None, None, None],
            textures["health_texture"],
            textures["smaller_empty_texture"],
        ).astype(float)
        info_pixels = _render_icons(info_pixels, health_texture, health_icon_locations)
        info_pixels = _render_two_digit_numbers(info_pixels, teammate_health, health_icon_locations)

        # Render teammate directions
        direction_icon_locations = player_icon_locations + jnp.array([0, 2])
        local_position = (
            state.player_position
            - state.player_position[player_index]
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < obs_dim_array
        ).all(axis=-1)  
        render_direction = jnp.logical_not(on_screen)

        direction_index_2d = jnp.where(
            local_position < 0, 0,
            jnp.where(local_position >= jnp.array([OBS_DIM[0], OBS_DIM[1]]), 2, 1)
        )
        direction_texture = textures["direction_textures"][
            direction_index_2d[:, 0], direction_index_2d[:, 1]
        ][:, :, :, :3]
        direction_texture = jax.vmap(jnp.multiply, in_axes=(0, 0))(
            direction_texture, render_direction
        ).astype(float)
        info_pixels = _render_icons(info_pixels, direction_texture, direction_icon_locations)
            

        return info_pixels

    teammate_info_pixels = jax.vmap(_render_teammate_info)(
        jnp.arange(static_params.player_count)
    )

    # Combine map and inventory
    pixels = jnp.concatenate([teammate_info_pixels, map_pixels, inv_pixels], axis=1)

    # # Downscale by 2
    # pixels = pixels[::downscale, ::downscale]

    return pixels


def render_craftax_text(state: EnvState):

    text_obs = "Map:\n"

    # Map
    map = state.map[state.player_level]

    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    # Map
    padded_grid = jnp.pad(
        map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )

    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2

    map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)

    padded_items_map = jnp.pad(
        state.item_map[state.player_level],
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=ItemType.NONE.value,
    )

    item_map_view = jax.lax.dynamic_slice(padded_items_map, tl_corner, OBS_DIM)

    # Mobs
    mob_types_per_class = 8
    mob_map = jnp.zeros(
        (*OBS_DIM, 5 * mob_types_per_class), dtype=jnp.int32
    )  # 5 classes * 8 types

    def _add_mob_to_map(carry, mob_index):
        mob_map, mobs, mob_class_index = carry

        local_position = (
            mobs.position[mob_index]
            - state.player_position
            + jnp.array([OBS_DIM[0], OBS_DIM[1]]) // 2
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < jnp.array([OBS_DIM[0], OBS_DIM[1]])
        ).all()
        on_screen *= mobs.mask[mob_index]

        mob_identifier = mob_class_index * mob_types_per_class + mobs.type_id[mob_index]
        mob_map = mob_map.at[local_position[0], local_position[1], mob_identifier].set(
            on_screen.astype(jnp.int32)
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

    def mob_id_to_name(id):
        if id == 0:
            return "Zombie"
        elif id == 1:
            return "Gnome Warrior"
        elif id == 2:
            return "Orc Soldier"
        elif id == 3:
            return "Lizard"
        elif id == 4:
            return "Knight"
        elif id == 5:
            return "Troll"
        elif id == 6:
            return "Pigman"
        elif id == 7:
            return "Frost Troll"
        elif id == 8:
            return "Cow"
        elif id == 9:
            return "Bat"
        elif id == 10:
            return "Snail"
        elif id == 16:
            return "Skeleton"
        elif id == 17:
            return "Gnome Archer"
        elif id == 18:
            return "Orc Mage"
        elif id == 19:
            return "Kobold"
        elif id == 20:
            return "Archer"
        elif id == 21:
            return "Deep Thing"
        elif id == 22:
            return "Fire Elemental"
        elif id == 23:
            return "Ice Elemental"
        elif id == 24:
            return "Arrow"
        elif id == 25:
            return "Dagger"
        elif id == 26:
            return "Fireball"
        elif id == 27:
            return "Iceball"
        elif id == 28:
            return "Arrow"
        elif id == 29:
            return "Slimeball"
        elif id == 30:
            return "Fireball"
        elif id == 31:
            return "Iceball"
        elif id == 32:
            return "Arrow (Player)"
        elif id == 33:
            return "Dagger (Player)"
        elif id == 34:
            return "Fireball (Player)"
        elif id == 35:
            return "Iceball (Player)"
        elif id == 36:
            return "Arrow (Player)"
        elif id == 37:
            return "Slimeball (Player)"
        elif id == 38:
            return "Fireball (Player)"
        elif id == 39:
            return "Iceball (Player)"

    padded_light_map = jnp.pad(
        state.light_map[state.player_level],
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=0.0,
    )
    light_map_view = jax.lax.dynamic_slice(padded_light_map, tl_corner, OBS_DIM) > 0.05

    for x in range(OBS_DIM[0]):
        for y in range(OBS_DIM[1]):
            text_obs += f"{y - OBS_DIM[1] // 2}, {x - OBS_DIM[0] // 2}: "
            if light_map_view[x, y]:
                if mob_map[x, y].max() > 0.5:
                    text_obs += mob_id_to_name(mob_map[x, y].argmax()) + " on "
                if item_map_view[x, y] != ItemType.NONE.value:
                    text_obs += ItemType(item_map_view[x, y]).name.lower() + " on "
                text_obs += BlockType(map_view[x, y]).name.lower() + "\n"
            else:
                text_obs += "Darkness\n"

    # Inventory
    text_obs += "\nInventory:\n"
    text_obs += f"Wood: {state.inventory.wood}\n"
    text_obs += f"Stone: {state.inventory.stone}\n"
    text_obs += f"Coal: {state.inventory.coal}\n"
    text_obs += f"Iron: {state.inventory.iron}\n"
    text_obs += f"Diamond: {state.inventory.diamond}\n"
    text_obs += f"Sapphire: {state.inventory.sapphire}\n"
    text_obs += f"Ruby: {state.inventory.ruby}\n"
    text_obs += f"Sapling: {state.inventory.sapling}\n"
    text_obs += f"Torch: {state.inventory.torches}\n"
    text_obs += f"Arrow: {state.inventory.arrows}\n"
    text_obs += f"Book: {state.inventory.books}\n"

    def level_to_material(level):
        if level == 1:
            return "Wood"
        elif level == 2:
            return "Stone"
        elif level == 3:
            return "Iron"
        elif level == 4:
            return "Diamond"

    def level_to_enchantment(level):
        if level == 0:
            return "No"
        if level == 1:
            return "Fire"
        elif level == 2:
            return "Ice"

    if state.inventory.pickaxe > 0:
        text_obs += level_to_material(state.inventory.pickaxe) + " Pickaxe\n"
    if state.inventory.sword > 0:
        text_obs += level_to_material(state.inventory.sword) + " Sword"
        text_obs += (
            " with " + level_to_enchantment(state.sword_enchantment) + " enchantment\n"
        )
    if state.inventory.bow > 0:
        text_obs += (
            "Bow with " + level_to_enchantment(state.bow_enchantment) + " enchantment\n"
        )

    text_obs += f"Red potion: {state.inventory.potions[0]}\n"
    text_obs += f"Green potion: {state.inventory.potions[1]}\n"
    text_obs += f"Blue potion: {state.inventory.potions[2]}\n"
    text_obs += f"Pink potion: {state.inventory.potions[3]}\n"
    text_obs += f"Cyan potion: {state.inventory.potions[4]}\n"
    text_obs += f"Yellow potion: {state.inventory.potions[5]}\n"

    def get_armour_level(level):
        if level == 1:
            return "Iron"
        elif level == 2:
            return "Diamond"

    if state.inventory.armour[0] > 0:
        text_obs += f"{get_armour_level(state.inventory.armour[0])} Helmet"
        text_obs += (
            " with "
            + level_to_enchantment(state.armour_enchantments[0])
            + " enchantment\n"
        )

    if state.inventory.armour[1] > 0:
        text_obs += f"{get_armour_level(state.inventory.armour[1])} Chestplate"
        text_obs += (
            " with "
            + level_to_enchantment(state.armour_enchantments[1])
            + " enchantment\n"
        )

    if state.inventory.armour[2] > 0:
        text_obs += f"{get_armour_level(state.inventory.armour[2])} Leggings"
        text_obs += (
            " with "
            + level_to_enchantment(state.armour_enchantments[2])
            + " enchantment\n"
        )

    if state.inventory.armour[3] > 0:
        text_obs += f"{get_armour_level(state.inventory.armour[3])} Boots"
        text_obs += (
            " with "
            + level_to_enchantment(state.armour_enchantments[3])
            + " enchantment\n"
        )

    text_obs += f"Health: {state.player_health}\n"
    text_obs += f"Food: {state.player_food}\n"
    text_obs += f"Drink: {state.player_drink}\n"
    text_obs += f"Energy: {state.player_energy}\n"
    text_obs += f"Mana: {state.player_mana}\n"
    text_obs += f"XP: {state.player_xp}\n"
    text_obs += f"Dexterity: {state.player_dexterity}\n"
    text_obs += f"Strength: {state.player_strength}\n"
    text_obs += f"Intelligence: {state.player_intelligence}\n"

    text_obs += f"Direction: {Action(state.player_direction).name.lower()}\n"

    text_obs += f"Light: {state.light_level}\n"
    text_obs += f"Is Sleeping: {state.is_sleeping}\n"
    text_obs += f"Is Resting: {state.is_resting}\n"
    text_obs += f"Learned Fireball: {state.learned_spells[0]}\n"
    text_obs += f"Learned Iceball: {state.learned_spells[1]}\n"
    text_obs += f"Floor: {state.player_level}\n"
    text_obs += f"Ladder Open: {state.monsters_killed[state.player_level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL}\n"
    text_obs += f"Is Boss Vulnerable: {is_boss_vulnerable(state)}\n"

    return text_obs
