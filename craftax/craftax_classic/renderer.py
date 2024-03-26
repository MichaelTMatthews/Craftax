from functools import partial

from craftax.craftax_classic.constants import *


def render_craftax_symbolic(state):
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    # Map
    padded_grid = jnp.pad(
        state.map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )

    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2

    map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)
    map_view_one_hot = jax.nn.one_hot(map_view, num_classes=len(BlockType))

    # Mobs
    mob_map = jnp.zeros((*OBS_DIM, 4), dtype=jnp.uint8)  # 4 types of mobs

    def _add_mob_to_map(carry, mob_index):
        mob_map, mobs, mob_type_index = carry

        local_position = (
            mobs.position[mob_index]
            - state.player_position
            + jnp.array([OBS_DIM[0], OBS_DIM[1]]) // 2
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < jnp.array([OBS_DIM[0], OBS_DIM[1]])
        ).all()
        on_screen *= mobs.mask[mob_index]

        mob_map = mob_map.at[local_position[0], local_position[1], mob_type_index].set(
            on_screen.astype(jnp.uint8)
        )

        return (mob_map, mobs, mob_type_index), None

    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (mob_map, state.zombies, 0),
        jnp.arange(state.zombies.mask.shape[0]),
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map, (mob_map, state.cows, 1), jnp.arange(state.cows.mask.shape[0])
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (mob_map, state.skeletons, 2),
        jnp.arange(state.skeletons.mask.shape[0]),
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (mob_map, state.arrows, 3),
        jnp.arange(state.arrows.mask.shape[0]),
    )

    all_map = jnp.concatenate([map_view_one_hot, mob_map], axis=-1)

    # Inventory
    inventory = (
        jnp.array(
            [
                state.inventory.wood,
                state.inventory.stone,
                state.inventory.coal,
                state.inventory.iron,
                state.inventory.diamond,
                state.inventory.sapling,
                state.inventory.wood_pickaxe,
                state.inventory.stone_pickaxe,
                state.inventory.iron_pickaxe,
                state.inventory.wood_sword,
                state.inventory.stone_sword,
                state.inventory.iron_sword,
            ]
        ).astype(jnp.float16)
        / 10.0
    )

    intrinsics = (
        jnp.array(
            [
                state.player_health,
                state.player_food,
                state.player_drink,
                state.player_energy,
            ]
        ).astype(jnp.float16)
        / 10.0
    )

    direction = jax.nn.one_hot(state.player_direction - 1, num_classes=4)

    all_flattened = jnp.concatenate(
        [
            all_map.flatten(),
            inventory,
            intrinsics,
            direction,
            jnp.array([state.light_level, state.is_sleeping]),
        ]
    )

    return all_flattened


@partial(jax.jit, static_argnums=(1,))
def render_craftax_pixels(state, block_pixel_size):
    textures = TEXTURES[block_pixel_size]
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    # RENDER MAP
    # Get view of map
    padded_grid = jnp.pad(
        state.map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )

    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2

    map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)

    # Render map tiles
    map_pixels_indexes = jnp.repeat(
        jnp.repeat(map_view, repeats=block_pixel_size, axis=0),
        repeats=block_pixel_size,
        axis=1,
    )
    map_pixels_indexes = jnp.expand_dims(map_pixels_indexes, axis=-1)
    map_pixels_indexes = jnp.repeat(map_pixels_indexes, repeats=3, axis=2)

    map_pixels = jnp.zeros(
        (OBS_DIM[0] * block_pixel_size, OBS_DIM[1] * block_pixel_size, 3),
        dtype=jnp.float32,
    )

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

    # Render player
    player_texture_index = jax.lax.select(
        state.is_sleeping, 4, state.player_direction - 1
    )
    map_pixels = (
        map_pixels
        * (1 - textures["full_map_player_textures_alpha"][player_texture_index])
        + textures["full_map_player_textures"][player_texture_index]
        * textures["full_map_player_textures_alpha"][player_texture_index]
    )

    # Render mobs
    # Zombies

    def _add_zombie_to_pixels(pixels, zombie_index):
        local_position = (
            state.zombies.position[zombie_index]
            - state.player_position
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < obs_dim_array
        ).all()
        on_screen *= state.zombies.mask[zombie_index]

        zombie_texture = textures["zombie_texture"] * on_screen

        zombie_texture_with_background = (
            1 - textures["zombie_texture_alpha"] * on_screen
        )

        zombie_texture_with_background = (
            zombie_texture_with_background
            * jax.lax.dynamic_slice(
                pixels,
                (
                    local_position[0] * block_pixel_size,
                    local_position[1] * block_pixel_size,
                    0,
                ),
                (block_pixel_size, block_pixel_size, 3),
            )
        )

        zombie_texture_with_background = (
            zombie_texture_with_background
            + zombie_texture * textures["zombie_texture_alpha"]
        )

        pixels = jax.lax.dynamic_update_slice(
            pixels,
            zombie_texture_with_background,
            (
                local_position[0] * block_pixel_size,
                local_position[1] * block_pixel_size,
                0,
            ),
        )

        return pixels, None

    map_pixels, _ = jax.lax.scan(
        _add_zombie_to_pixels, map_pixels, jnp.arange(state.zombies.mask.shape[0])
    )

    def _add_cow_to_pixels(pixels, cow_index):
        local_position = (
            state.cows.position[cow_index]
            - state.player_position
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < obs_dim_array
        ).all()
        on_screen *= state.cows.mask[cow_index]

        cow_texture = textures["cow_texture"] * on_screen

        cow_texture_with_background = 1 - textures["cow_texture_alpha"] * on_screen

        cow_texture_with_background = (
            cow_texture_with_background
            * jax.lax.dynamic_slice(
                pixels,
                (
                    local_position[0] * block_pixel_size,
                    local_position[1] * block_pixel_size,
                    0,
                ),
                (block_pixel_size, block_pixel_size, 3),
            )
        )

        cow_texture_with_background = (
            cow_texture_with_background + cow_texture * textures["cow_texture_alpha"]
        )

        pixels = jax.lax.dynamic_update_slice(
            pixels,
            cow_texture_with_background,
            (
                local_position[0] * block_pixel_size,
                local_position[1] * block_pixel_size,
                0,
            ),
        )

        return pixels, None

    map_pixels, _ = jax.lax.scan(
        _add_cow_to_pixels, map_pixels, jnp.arange(state.cows.mask.shape[0])
    )

    def _add_skeleton_to_pixels(pixels, skeleton_index):
        local_position = (
            state.skeletons.position[skeleton_index]
            - state.player_position
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < obs_dim_array
        ).all()
        on_screen *= state.skeletons.mask[skeleton_index]

        skeleton_texture = textures["skeleton_texture"] * on_screen

        skeleton_texture_with_background = (
            1 - textures["skeleton_texture_alpha"] * on_screen
        )

        skeleton_texture_with_background = (
            skeleton_texture_with_background
            * jax.lax.dynamic_slice(
                pixels,
                (
                    local_position[0] * block_pixel_size,
                    local_position[1] * block_pixel_size,
                    0,
                ),
                (block_pixel_size, block_pixel_size, 3),
            )
        )

        skeleton_texture_with_background = (
            skeleton_texture_with_background
            + skeleton_texture * textures["skeleton_texture_alpha"]
        )

        pixels = jax.lax.dynamic_update_slice(
            pixels,
            skeleton_texture_with_background,
            (
                local_position[0] * block_pixel_size,
                local_position[1] * block_pixel_size,
                0,
            ),
        )

        return pixels, None

    map_pixels, _ = jax.lax.scan(
        _add_skeleton_to_pixels, map_pixels, jnp.arange(state.skeletons.mask.shape[0])
    )

    def _add_arrow_to_pixels(pixels, arrow_index):
        local_position = (
            state.arrows.position[arrow_index]
            - state.player_position
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < obs_dim_array
        ).all()
        on_screen *= state.arrows.mask[arrow_index]

        arrow_texture = textures["arrow_texture"]
        arrow_texture_alpha = textures["arrow_texture_alpha"]

        flipped_arrow_texture = jnp.flip(arrow_texture, axis=0)
        flipped_arrow_texture_alpha = jnp.flip(arrow_texture_alpha, axis=0)
        flip_arrow = jnp.logical_or(
            state.arrow_directions[arrow_index, 0] > 0,
            state.arrow_directions[arrow_index, 1] > 0,
        )

        arrow_texture = jax.lax.select(
            flip_arrow,
            flipped_arrow_texture,
            arrow_texture,
        )
        arrow_texture_alpha = jax.lax.select(
            flip_arrow,
            flipped_arrow_texture_alpha,
            arrow_texture_alpha,
        )

        transposed_arrow_texture = jnp.transpose(arrow_texture, (1, 0, 2))
        transposed_arrow_texture_alpha = jnp.transpose(arrow_texture_alpha, (1, 0, 2))

        arrow_texture = jax.lax.select(
            state.arrow_directions[arrow_index, 1] != 0,
            transposed_arrow_texture,
            arrow_texture,
        )
        arrow_texture_alpha = jax.lax.select(
            state.arrow_directions[arrow_index, 1] != 0,
            transposed_arrow_texture_alpha,
            arrow_texture_alpha,
        )

        arrow_texture = arrow_texture * on_screen
        arrow_texture_with_background = 1 - arrow_texture_alpha * on_screen

        arrow_texture_with_background = (
            arrow_texture_with_background
            * jax.lax.dynamic_slice(
                pixels,
                (
                    local_position[0] * block_pixel_size,
                    local_position[1] * block_pixel_size,
                    0,
                ),
                (block_pixel_size, block_pixel_size, 3),
            )
        )

        arrow_texture_with_background = (
            arrow_texture_with_background + arrow_texture * arrow_texture_alpha
        )

        pixels = jax.lax.dynamic_update_slice(
            pixels,
            arrow_texture_with_background,
            (
                local_position[0] * block_pixel_size,
                local_position[1] * block_pixel_size,
                0,
            ),
        )

        return pixels, None

    map_pixels, _ = jax.lax.scan(
        _add_arrow_to_pixels, map_pixels, jnp.arange(state.arrows.mask.shape[0])
    )

    # Apply night
    daylight = state.light_level

    night_static_intensity = 2 * (0.5 - daylight)
    night_static_intensity = jnp.maximum(night_static_intensity, 0.0)
    night_with_static = (
        jax.random.uniform(state.state_rng, map_pixels.shape[:2]) * 95 + 32
    )
    night_static_mask = (
        night_static_intensity * textures["night_noise_intensity_texture"]
    )
    night_with_static = (
        1 - night_static_mask
    ) * map_pixels + night_static_mask * night_with_static[:, :, None]

    night_pixels = jax.lax.select(daylight < 0.5, night_with_static, map_pixels)

    # Enhance
    enhance_factor = 0.4
    lum = (
        0.299 * night_pixels[:, :, 0]
        + 0.587 * night_pixels[:, :, 1]
        + 0.114 * night_pixels[:, :, 2]
    )
    lum = jnp.expand_dims(lum, axis=-1).repeat(3, axis=-1)
    night_pixels = night_pixels * enhance_factor + (1 - enhance_factor) * lum

    # Tint
    night_pixels = 0.5 * night_pixels + 0.5 * textures["night_texture"]

    # Blend with map pixels
    map_pixels = daylight * map_pixels + (1 - daylight) * night_pixels

    # Apply sleep
    sleep_pixels = (
        0.299 * map_pixels[:, :, 0]
        + 0.587 * map_pixels[:, :, 1]
        + 0.114 * map_pixels[:, :, 2]
    )
    sleep_pixels = (0.5 * sleep_pixels)[:, :, None] + (0.5 * jnp.array([0, 0, 16]))[
        None, None, :
    ]
    map_pixels = (1 - state.is_sleeping) * map_pixels + state.is_sleeping * sleep_pixels

    # Render mob map
    # mob_map_pixels = (
    #     jnp.array([[[128, 0, 0]]]).repeat(OBS_DIM, axis=0).repeat(OBS_DIM, axis=1)
    # )
    # padded_mob_map = jnp.pad(state.mob_map, OBS_DIM // 2 + 2, constant_values=False)
    # mob_map_view = jax.lax.dynamic_slice(padded_mob_map, tl_corner, (OBS_DIM, OBS_DIM))
    # mob_map_pixels = mob_map_pixels * jnp.expand_dims(mob_map_view, axis=-1)
    # mob_map_pixels = mob_map_pixels.repeat(block_pixel_size, axis=0).repeat(
    #     block_pixel_size, axis=1
    # )
    # map_pixels = map_pixels + mob_map_pixels

    # RENDER INVENTORY
    inv_pixel_left_space = (block_pixel_size - int(0.8 * block_pixel_size)) // 2 - 1
    inv_pixel_right_space = (
        block_pixel_size - int(0.8 * block_pixel_size) - inv_pixel_left_space
    )

    inv_pixels = jnp.zeros(
        (INVENTORY_OBS_HEIGHT * block_pixel_size, OBS_DIM[1] * block_pixel_size, 3),
        dtype=jnp.float32,
    )

    number_size = int(block_pixel_size * 0.6)
    number_offset = block_pixel_size - number_size

    def _render_number(pixels, number, x, y):
        pixels = pixels.at[
            y * block_pixel_size + number_offset - 1 : (y + 1) * block_pixel_size - 1,
            x * block_pixel_size + number_offset - 1 : (x + 1) * block_pixel_size - 1,
        ].mul(1 - textures["number_textures_alpha"][number])

        pixels = pixels.at[
            y * block_pixel_size + number_offset - 1 : (y + 1) * block_pixel_size - 1,
            x * block_pixel_size + number_offset - 1 : (x + 1) * block_pixel_size - 1,
        ].add(textures["number_textures"][number])

        return pixels

    # Render player stats
    health_texture = jax.lax.select(
        state.player_health > 0,
        textures["health_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
    ].set(health_texture)
    inv_pixels = _render_number(inv_pixels, state.player_health, 0, 0)

    hunger_texture = jax.lax.select(
        state.player_food > 0,
        textures["hunger_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
        block_pixel_size
        + inv_pixel_left_space : block_pixel_size * 2
        - inv_pixel_right_space,
    ].set(hunger_texture)
    inv_pixels = _render_number(inv_pixels, state.player_food, 1, 0)

    thirst_texture = jax.lax.select(
        state.player_drink > 0,
        textures["thirst_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
        block_pixel_size * 2
        + inv_pixel_left_space : block_pixel_size * 3
        - inv_pixel_right_space,
    ].set(thirst_texture)
    inv_pixels = _render_number(inv_pixels, state.player_drink, 2, 0)

    energy_texture = jax.lax.select(
        state.player_energy > 0,
        textures["energy_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
        block_pixel_size * 3
        + inv_pixel_left_space : block_pixel_size * 4
        - inv_pixel_right_space,
    ].set(energy_texture)
    inv_pixels = _render_number(inv_pixels, state.player_energy, 3, 0)

    # Render inventory

    inv_wood_texture = jax.lax.select(
        state.inventory.wood > 0,
        textures["smaller_block_textures"][BlockType.WOOD.value],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
        block_pixel_size * 5
        + inv_pixel_left_space : block_pixel_size * 6
        - inv_pixel_right_space,
    ].set(inv_wood_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.wood, 5, 0)

    inv_stone_texture = jax.lax.select(
        state.inventory.stone > 0,
        textures["smaller_block_textures"][BlockType.STONE.value],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
        block_pixel_size * 6
        + inv_pixel_left_space : block_pixel_size * 7
        - inv_pixel_right_space,
    ].set(inv_stone_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.stone, 6, 0)

    inv_coal_texture = jax.lax.select(
        state.inventory.coal > 0,
        textures["smaller_block_textures"][BlockType.COAL.value],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
        block_pixel_size * 7
        + inv_pixel_left_space : block_pixel_size * 8
        - inv_pixel_right_space,
    ].set(inv_coal_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.coal, 7, 0)

    inv_iron_texture = jax.lax.select(
        state.inventory.iron > 0,
        textures["smaller_block_textures"][BlockType.IRON.value],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
        block_pixel_size * 8
        + inv_pixel_left_space : block_pixel_size * 9
        - inv_pixel_right_space,
    ].set(inv_iron_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.iron, 8, 0)

    inv_diamond_texture = jax.lax.select(
        state.inventory.diamond > 0,
        textures["smaller_block_textures"][BlockType.DIAMOND.value],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        block_pixel_size
        + inv_pixel_left_space : 2 * block_pixel_size
        - inv_pixel_right_space,
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
    ].set(inv_diamond_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.diamond, 0, 1)

    inv_sapling_texture = jax.lax.select(
        state.inventory.sapling > 0,
        textures["sapling_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
        block_pixel_size * 4
        + inv_pixel_left_space : block_pixel_size * 5
        - inv_pixel_right_space,
    ].set(inv_sapling_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.sapling, 4, 0)

    # Render tools
    # Wooden pickaxe
    wooden_pickaxe_maybe_texture = jax.lax.select(
        state.inventory.wood_pickaxe > 0,
        textures["wood_pickaxe_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        block_pixel_size
        + inv_pixel_left_space : block_pixel_size * 2
        - inv_pixel_right_space,
        block_pixel_size
        + inv_pixel_left_space : block_pixel_size * 2
        - inv_pixel_right_space,
    ].set(wooden_pickaxe_maybe_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.wood_pickaxe, 1, 1)

    # Stone pickaxe
    stone_pickaxe_maybe_texture = jax.lax.select(
        state.inventory.stone_pickaxe > 0,
        textures["stone_pickaxe_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        block_pixel_size
        + inv_pixel_left_space : block_pixel_size * 2
        - inv_pixel_right_space,
        block_pixel_size * 2
        + inv_pixel_left_space : block_pixel_size * 3
        - inv_pixel_right_space,
    ].set(stone_pickaxe_maybe_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.stone_pickaxe, 2, 1)

    # Iron pickaxe
    iron_pickaxe_maybe_texture = jax.lax.select(
        state.inventory.iron_pickaxe > 0,
        textures["iron_pickaxe_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        block_pixel_size
        + inv_pixel_left_space : block_pixel_size * 2
        - inv_pixel_right_space,
        block_pixel_size * 3
        + inv_pixel_left_space : block_pixel_size * 4
        - inv_pixel_right_space,
    ].set(iron_pickaxe_maybe_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.iron_pickaxe, 3, 1)

    # Wooden sword
    wooden_sword_maybe_texture = jax.lax.select(
        state.inventory.wood_sword > 0,
        textures["wood_sword_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        block_pixel_size
        + inv_pixel_left_space : block_pixel_size * 2
        - inv_pixel_right_space,
        block_pixel_size * 4
        + inv_pixel_left_space : block_pixel_size * 5
        - inv_pixel_right_space,
    ].set(wooden_sword_maybe_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.wood_sword, 4, 1)

    # Stone sword
    stone_sword_maybe_texture = jax.lax.select(
        state.inventory.stone_sword > 0,
        textures["stone_sword_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        block_pixel_size
        + inv_pixel_left_space : block_pixel_size * 2
        - inv_pixel_right_space,
        block_pixel_size * 5
        + inv_pixel_left_space : block_pixel_size * 6
        - inv_pixel_right_space,
    ].set(stone_sword_maybe_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.stone_sword, 5, 1)

    # Iron sword
    iron_sword_maybe_texture = jax.lax.select(
        state.inventory.iron_sword > 0,
        textures["iron_sword_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        block_pixel_size
        + inv_pixel_left_space : block_pixel_size * 2
        - inv_pixel_right_space,
        block_pixel_size * 6
        + inv_pixel_left_space : block_pixel_size * 7
        - inv_pixel_right_space,
    ].set(iron_sword_maybe_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.iron_sword, 6, 1)

    # Combine map and inventory
    pixels = jnp.concatenate([map_pixels, inv_pixels], axis=0)

    # # Downscale by 2
    # pixels = pixels[::downscale, ::downscale]

    return pixels


def render_pixels_empty(block_pixel_size):
    pixels = jnp.zeros(
        (
            OBS_DIM * block_pixel_size,
            (OBS_DIM + INVENTORY_OBS_HEIGHT) * block_pixel_size,
            3,
        ),
        dtype=jnp.float32,
    )
    return pixels
