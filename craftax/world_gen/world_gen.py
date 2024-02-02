import jax
import jax.scipy as jsp

from ..constants import *
from ..game_logic import calculate_light_level, get_distance_map
from ..craftax_state import EnvState, Inventory, Mobs
from ..util.noise import generate_fractal_noise_2d
from ..world_gen.world_gen_configs import (
    OVERWORLD_CONFIG,
    GNOMISH_MINES_CONFIG,
    FIRE_LEVEL_CONFIG,
    ICE_LEVEL_CONFIG,
    BOSS_LEVEL_CONFIG,
    ALL_DUNGEON_CONFIGS,
    ALL_SMOOTHGEN_CONFIGS,
)


def dungeon_precompute(rng, static_params):
    pass


def generate_dungeon(rng, static_params, config):

    chunk_size = 16
    world_chunk_width = static_params.map_size[0] // chunk_size
    world_chunk_height = static_params.map_size[1] // chunk_size
    room_occupancy_chunks = jnp.ones(world_chunk_width * world_chunk_height)

    num_rooms = 8
    min_room_size = 5
    max_room_size = 10

    rng, _rng, __rng = jax.random.split(rng, 3)
    room_sizes = jax.random.randint(
        __rng, shape=(num_rooms, 2), minval=min_room_size, maxval=max_room_size
    )

    map = jnp.ones(static_params.map_size, dtype=jnp.int32) * BlockType.WALL.value
    padded_map = jnp.pad(map, max_room_size, constant_values=0)

    def _add_room(carry, room_index):
        cmap, room_occupancy_chunks, rng = carry

        rng, _rng = jax.random.split(rng)
        room_chunk = jax.random.choice(
            _rng,
            jnp.arange(world_chunk_width * world_chunk_height),
            p=room_occupancy_chunks,
        )
        room_occupancy_chunks = room_occupancy_chunks.at[room_chunk].set(0)

        room_position = jnp.array(
            [
                (room_chunk % world_chunk_height) * chunk_size,
                (room_chunk // world_chunk_height) * chunk_size,
            ]
        ) + jnp.array([max_room_size, max_room_size])
        rng, _rng = jax.random.split(rng)
        room_position += jax.random.randint(
            _rng, (2,), minval=0, maxval=chunk_size - min_room_size
        )

        slice = jax.lax.dynamic_slice(
            cmap, room_position, (max_room_size, max_room_size)
        )
        xs = jnp.expand_dims(jnp.arange(max_room_size), axis=-1).repeat(
            max_room_size, axis=-1
        )
        ys = jnp.expand_dims(jnp.arange(max_room_size), axis=0).repeat(
            max_room_size, axis=0
        )

        room_mask = jnp.logical_and(
            xs < room_sizes[room_index, 0], ys < room_sizes[room_index, 1]
        )

        slice = room_mask * BlockType.PATH.value + (1 - room_mask) * slice

        cmap = jax.lax.dynamic_update_slice(
            cmap,
            slice,
            room_position,
        )

        # Torches in corner
        cmap = cmap.at[room_position[0], room_position[1]].set(
            BlockType.TORCH_ON_PATH.value
        )
        cmap = cmap.at[
            room_position[0] + room_sizes[room_index, 0] - 1, room_position[1]
        ].set(BlockType.TORCH_ON_PATH.value)
        cmap = cmap.at[
            room_position[0], room_position[1] + room_sizes[room_index, 1] - 1
        ].set(BlockType.TORCH_ON_PATH.value)
        cmap = cmap.at[
            room_position[0] + room_sizes[room_index, 0] - 1,
            room_position[1] + room_sizes[room_index, 1] - 1,
        ].set(BlockType.TORCH_ON_PATH.value)

        # Chest
        rng, _rng = jax.random.split(rng)
        chest_position = jax.random.randint(
            _rng, shape=(2,), minval=jnp.zeros(2), maxval=room_sizes[room_index]
        )
        cmap = cmap.at[
            room_position[0] + chest_position[0], room_position[1] + chest_position[1]
        ].set(BlockType.CHEST.value)

        # Fountain
        rng, _rng, __rng = jax.random.split(rng, 3)
        fountain_position = jax.random.randint(
            _rng, shape=(2,), minval=jnp.zeros(2), maxval=room_sizes[room_index]
        )
        room_has_fountain = jax.random.uniform(__rng) > 0.5
        fountain_block = (
            room_has_fountain * config.fountain_block
            + (1 - room_has_fountain)
            * cmap[
                room_position[0] + fountain_position[0],
                room_position[1] + fountain_position[1],
            ]
        )
        cmap = cmap.at[
            room_position[0] + fountain_position[0],
            room_position[1] + fountain_position[1],
        ].set(fountain_block)

        return (cmap, room_occupancy_chunks, rng), room_position

    rng, _rng = jax.random.split(rng)
    (padded_map, _, _), room_positions = jax.lax.scan(
        _add_room, (padded_map, room_occupancy_chunks, _rng), jnp.arange(num_rooms)
    )

    def _add_path(carry, path_index):
        cmap, included_rooms_mask, rng = carry

        path_source = room_positions[path_index]

        rng, _rng = jax.random.split(rng)
        sink_index = jax.random.choice(
            _rng, jnp.arange(num_rooms), p=included_rooms_mask
        )
        path_sink = room_positions[sink_index]

        # Horizontal component
        entire_row = cmap[path_source[0]]
        path_indexes = jnp.arange(static_params.map_size[0] + 2 * max_room_size)
        path_indexes = path_indexes - path_source[1]
        horizontal_distance = path_sink[1] - path_source[1]
        path_indexes = path_indexes * jnp.sign(horizontal_distance)

        horizontal_mask = jnp.logical_and(
            path_indexes >= 0, path_indexes <= jnp.abs(horizontal_distance)
        )
        horizontal_mask = jnp.logical_and(
            horizontal_mask, jnp.sign(horizontal_distance)
        )
        horizontal_mask = jnp.logical_and(
            horizontal_mask, entire_row == BlockType.WALL.value
        )

        new_row = (
            horizontal_mask * BlockType.PATH.value + (1 - horizontal_mask) * entire_row
        )

        cmap = jax.lax.dynamic_update_slice(
            cmap,
            jnp.expand_dims(new_row, axis=0),
            path_source,
        )

        # Vertical component
        entire_col = cmap[:, path_sink[1]]
        path_indexes = jnp.arange(static_params.map_size[1] + 2 * max_room_size)
        path_indexes = path_indexes - path_source[0]
        vertical_distance = path_sink[0] - path_source[0]
        path_indexes = path_indexes * jnp.sign(vertical_distance)

        vertical_mask = jnp.logical_and(
            path_indexes >= 0, path_indexes <= jnp.abs(vertical_distance)
        )
        vertical_mask = jnp.logical_and(vertical_mask, jnp.sign(vertical_distance))

        vertical_mask = jnp.logical_and(
            vertical_mask, entire_col == BlockType.WALL.value
        )

        new_col = (
            vertical_mask * BlockType.PATH.value + (1 - vertical_mask) * entire_col
        )

        cmap = jax.lax.dynamic_update_slice(
            cmap,
            jnp.expand_dims(new_col, axis=-1),
            path_sink,
        )

        rng, _rng = jax.random.split(rng)
        included_rooms_mask = included_rooms_mask.at[path_index].set(True)
        return (cmap, included_rooms_mask, _rng), None

    rng, _rng = jax.random.split(rng)
    included_rooms_mask = jnp.zeros(num_rooms, dtype=bool).at[-1].set(True)
    (padded_map, _, _), _, = jax.lax.scan(
        _add_path, (padded_map, included_rooms_mask, _rng), jnp.arange(0, num_rooms)
    )

    # Place special block in a random room
    special_block_position = room_positions[0] + jnp.array([2, 2])
    padded_map = padded_map.at[
        special_block_position[0], special_block_position[1]
    ].set(config.special_block)

    map = padded_map[max_room_size:-max_room_size, max_room_size:-max_room_size]

    # Visual stuff
    c_path_map = map != BlockType.WALL.value
    z = jnp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    adj_path_map = jsp.signal.convolve(c_path_map, z, mode="same")
    adj_path_map = adj_path_map > 0.5

    rng, _rng = jax.random.split(rng)
    rare_map = jax.random.choice(
        _rng,
        jnp.array([False, True]),
        static_params.map_size,
        p=jnp.array([0.9, 0.1]),
    )

    wall_map = (
        rare_map * BlockType.WALL_MOSS.value + (1 - rare_map) * BlockType.WALL.value
    )

    rare_map = jnp.logical_and(rare_map, map == BlockType.PATH.value)
    path_map = rare_map * config.rare_path_replacement_block + (1 - rare_map) * map

    is_wall_map = jnp.logical_and(map == BlockType.WALL.value, adj_path_map)
    is_darkness_map = jnp.logical_not(adj_path_map)
    is_path_map = jnp.logical_not(jnp.logical_or(is_wall_map, is_darkness_map))

    map = (
        is_path_map * path_map
        + is_wall_map * wall_map
        + is_darkness_map * BlockType.DARKNESS.value
    )

    light_map = jnp.ones(static_params.map_size, dtype=jnp.float32)

    # Ladders
    valid_ladder_down = (map.flatten() == BlockType.PATH.value).astype(jnp.float32)
    rng, _rng = jax.random.split(rng)
    ladder_index = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        p=valid_ladder_down / valid_ladder_down.sum(),
    )
    ladder_down_position = jnp.array(
        [
            ladder_index // static_params.map_size[0],
            ladder_index % static_params.map_size[0],
        ]
    )

    map = map.at[ladder_down_position[0], ladder_down_position[1]].set(
        BlockType.LADDER_DOWN.value
    )

    valid_ladder_up = map.flatten() == BlockType.PATH.value
    rng, _rng = jax.random.split(rng)
    ladder_index = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        p=valid_ladder_up,
    )
    ladder_up_position = jnp.array(
        [
            ladder_index // static_params.map_size[0],
            ladder_index % static_params.map_size[0],
        ]
    )
    map = map.at[ladder_up_position[0], ladder_up_position[1]].set(
        BlockType.LADDER_UP.value
    )

    return map, light_map, ladder_down_position, ladder_up_position


def smoothworld_precompute(rng, static_params):
    larger_res = (static_params.map_size[0] // 4, static_params.map_size[1] // 4)
    small_res = (static_params.map_size[0] // 16, static_params.map_size[1] // 16)
    x_res = (static_params.map_size[0] // 8, static_params.map_size[1] // 2)

    rng, _rng = jax.random.split(rng)
    water = generate_fractal_noise_2d(
        _rng, static_params.map_size, small_res, octaves=1
    )

    rng, _rng = jax.random.split(rng)
    mountain = (
        generate_fractal_noise_2d(_rng, static_params.map_size, small_res, octaves=1)
        + 0.05
    )

    rng, _rng = jax.random.split(rng)
    path_x = generate_fractal_noise_2d(_rng, static_params.map_size, x_res, octaves=1)

    tree_noise = generate_fractal_noise_2d(
        _rng, static_params.map_size, larger_res, octaves=1
    )

    return water, mountain, path_x, tree_noise


def generate_smoothworld(rng, static_params, player_position, config):
    player_proximity_map = get_distance_map(
        player_position, static_params.map_size
    ).astype(jnp.float32)
    player_proximity_map_water = (
        player_proximity_map / config.player_proximity_map_water_strength
    )
    player_proximity_map_water = jnp.clip(
        player_proximity_map_water, 0.0, config.player_proximity_map_water_max
    )

    player_proximity_map_mountain = (
        player_proximity_map / config.player_proximity_map_mountain_strength
    )
    player_proximity_map_mountain = jnp.clip(
        player_proximity_map_mountain,
        0.0,
        config.player_proximity_map_mountain_max,
    )

    larger_res = (static_params.map_size[0] // 4, static_params.map_size[1] // 4)
    small_res = (static_params.map_size[0] // 16, static_params.map_size[1] // 16)
    x_res = (static_params.map_size[0] // 8, static_params.map_size[1] // 2)

    rng, _rng = jax.random.split(rng)
    water = generate_fractal_noise_2d(
        _rng, static_params.map_size, small_res, octaves=1
    )
    water = water + player_proximity_map_water - 1.0

    # Water
    rng, _rng = jax.random.split(rng)
    map = jnp.where(
        water > config.water_threshold, config.sea_block, config.default_block
    )

    sand_map = jnp.logical_and(
        water > config.sand_threshold,
        map != config.sea_block,
    )

    map = jnp.where(sand_map, config.coast_block, map)

    # Mountain vs grass
    mountain_threshold = 0.7

    rng, _rng = jax.random.split(rng)
    mountain = (
        generate_fractal_noise_2d(_rng, static_params.map_size, small_res, octaves=1)
        + 0.05
    )
    mountain = mountain + player_proximity_map_mountain - 1.0
    map = jnp.where(mountain > mountain_threshold, config.mountain_block, map)

    # Paths
    rng, _rng = jax.random.split(rng)
    path_x = generate_fractal_noise_2d(_rng, static_params.map_size, x_res, octaves=1)
    path = jnp.logical_and(mountain > mountain_threshold, path_x > 0.8)
    map = jnp.where(path > 0.5, config.path_block, map)

    path_y = path_x.T
    path = jnp.logical_and(mountain > mountain_threshold, path_y > 0.8)
    map = jnp.where(path > 0.5, config.path_block, map)

    # Caves
    rng, _rng = jax.random.split(rng)
    caves = jnp.logical_and(mountain > 0.85, water > 0.4)
    map = jnp.where(caves > 0.5, config.inner_mountain_block, map)

    # Trees
    rng, _rng = jax.random.split(rng)
    tree_noise = generate_fractal_noise_2d(
        _rng, static_params.map_size, larger_res, octaves=1
    )
    tree = (tree_noise > config.tree_threshold_perlin) * jax.random.uniform(
        rng, shape=static_params.map_size
    ) > config.tree_threshold_uniform
    tree = jnp.logical_and(tree, map == config.tree_requirement_block)
    map = jnp.where(tree, config.tree, map)

    # Ores
    def _add_ore(carry, index):
        rng, map = carry
        rng, _rng = jax.random.split(rng)
        ore_map = jnp.logical_and(
            map == config.ore_requirement_blocks[index],
            jax.random.uniform(_rng, static_params.map_size)
            < config.ore_chances[index],
        )
        map = jnp.where(ore_map, config.ores[index], map)

        return (rng, map), None

    rng, _rng = jax.random.split(rng)
    (_, map), _ = jax.lax.scan(_add_ore, (_rng, map), jnp.arange(5))

    # Lava
    lava_map = jnp.logical_and(
        mountain > 0.85,
        tree_noise > 0.7,
    )
    map = jnp.where(lava_map, config.lava, map)

    # Make sure player spawns on grass
    map = map.at[player_position[0], player_position[1]].set(config.player_spawn)

    light_map = (
        jnp.ones(static_params.map_size, dtype=jnp.float32) * config.default_light
    )

    valid_ladder_down = map.flatten() == config.valid_ladder
    rng, _rng = jax.random.split(rng)
    ladder_index = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        p=valid_ladder_down,
    )
    ladder_down = jnp.array(
        [
            ladder_index // static_params.map_size[0],
            ladder_index % static_params.map_size[0],
        ]
    )

    map = map.at[ladder_down[0], ladder_down[1]].set(
        BlockType.LADDER_DOWN.value * config.ladder_down
        + map[ladder_down[0], ladder_down[1]] * (1 - config.ladder_down)
    )

    valid_ladder_up = map.flatten() == config.valid_ladder
    rng, _rng = jax.random.split(rng)
    ladder_index = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        p=valid_ladder_up,
    )
    ladder_up = jnp.array(
        [
            ladder_index // static_params.map_size[0],
            ladder_index % static_params.map_size[0],
        ]
    )

    LIGHT_MAP_AROUND_LADDER = TORCH_LIGHT_MAP * (
        1 - config.default_light
    ) + config.default_light * jnp.ones((9, 9))

    light_map = jax.lax.dynamic_update_slice(
        light_map, LIGHT_MAP_AROUND_LADDER, ladder_up - jnp.array([4, 4])
    )

    map = map.at[ladder_up[0], ladder_up[1]].set(
        BlockType.LADDER_UP.value * config.ladder_up
        + map[ladder_up[0], ladder_up[1]] * (1 - config.ladder_up)
    )

    return map, light_map, ladder_down, ladder_up


def generate_world(rng, params, static_params):

    player_position = jnp.array(
        [static_params.map_size[0] // 2, static_params.map_size[1] // 2]
    )

    rngs = jax.random.split(rng, 7)
    rng, _rng = rngs[0], rngs[1:]
    smoothgens = jax.vmap(generate_smoothworld, in_axes=(0, None, None, 0))(
        _rng, static_params, player_position, ALL_SMOOTHGEN_CONFIGS
    )

    rngs = jax.random.split(rng, 4)
    rng, _rng = rngs[0], rngs[1:]
    dungeons = jax.vmap(generate_dungeon, in_axes=(0, None, 0))(
        _rng, static_params, ALL_DUNGEON_CONFIGS
    )

    map = jnp.stack(
        (
            smoothgens[0][0],
            dungeons[0][0],
            smoothgens[0][1],
            dungeons[0][1],
            dungeons[0][2],
            smoothgens[0][2],
            smoothgens[0][3],
            smoothgens[0][4],
            smoothgens[0][5],
        ),
        axis=0,
    )
    light_map = jnp.stack(
        (
            smoothgens[1][0],
            dungeons[1][0],
            smoothgens[1][1],
            dungeons[1][1],
            dungeons[1][2],
            smoothgens[1][2],
            smoothgens[1][3],
            smoothgens[1][4],
            smoothgens[1][5],
        ),
        axis=0,
    )

    ladders_down = jnp.stack(
        (
            smoothgens[2][0],
            dungeons[2][0],
            smoothgens[2][1],
            dungeons[2][1],
            dungeons[2][2],
            smoothgens[2][2],
            smoothgens[2][3],
            smoothgens[2][4],
            smoothgens[2][5],
        ),
        axis=0,
    )

    ladders_up = jnp.stack(
        (
            smoothgens[3][0],
            dungeons[3][0],
            smoothgens[3][1],
            dungeons[3][1],
            dungeons[3][2],
            smoothgens[3][2],
            smoothgens[3][3],
            smoothgens[3][4],
            smoothgens[3][5],
        ),
        axis=0,
    )

    # map = jnp.expand_dims(map1, axis=0)
    # light_map = jnp.expand_dims(light_map1, axis=0)
    # ladders_down = jnp.expand_dims(ladder_down1, axis=0)
    # ladders_up = jnp.expand_dims(ladder_down1, axis=0)

    # Zombies

    z_pos = jnp.zeros(
        (static_params.num_levels, static_params.max_melee_mobs, 2), dtype=jnp.int32
    )
    z_health = jnp.ones(
        (static_params.num_levels, static_params.max_melee_mobs), dtype=jnp.float32
    )
    z_mask = jnp.zeros(
        (static_params.num_levels, static_params.max_melee_mobs), dtype=bool
    )

    # z_pos = z_pos.at[0, 0].set(player_position + jnp.array([1, 0]))
    # z_mask = z_mask.at[0, 0].set(True)
    # z_pos = z_pos.at[1].set(player_position + jnp.array([2, 0]))
    # z_mask = z_mask.at[1].set(True)

    melee_mobs = Mobs(
        position=z_pos,
        health=z_health,
        mask=z_mask,
        attack_cooldown=jnp.zeros(
            (static_params.num_levels, static_params.max_melee_mobs), dtype=jnp.int32
        ),
        type_id=jnp.expand_dims(jnp.arange(static_params.num_levels), axis=-1).repeat(
            static_params.max_melee_mobs, axis=-1
        ),
    )

    # Skeletons
    sk_positions = jnp.zeros(
        (static_params.num_levels, static_params.max_ranged_mobs, 2), dtype=jnp.int32
    )
    sk_healths = jnp.zeros(
        (static_params.num_levels, static_params.max_ranged_mobs), dtype=jnp.float32
    )
    sk_mask = jnp.zeros(
        (static_params.num_levels, static_params.max_ranged_mobs), dtype=bool
    )

    ranged_mobs = Mobs(
        position=sk_positions,
        health=sk_healths,
        mask=sk_mask,
        attack_cooldown=jnp.zeros(
            (static_params.num_levels, static_params.max_ranged_mobs), dtype=jnp.int32
        ),
        type_id=jnp.expand_dims(jnp.arange(static_params.num_levels), axis=-1).repeat(
            static_params.max_ranged_mobs, axis=-1
        ),
    )

    # Projectiles
    def _create_projectiles(max_num):
        projectile_positions = jnp.zeros(
            (static_params.num_levels, max_num, 2), dtype=jnp.int32
        )
        projectile_healths = jnp.zeros(
            (static_params.num_levels, max_num), dtype=jnp.int32
        )
        projectile_masks = jnp.zeros((static_params.num_levels, max_num), dtype=bool)

        projectiles = Mobs(
            position=projectile_positions,
            health=projectile_healths,
            mask=projectile_masks,
            attack_cooldown=jnp.zeros(
                (static_params.num_levels, max_num), dtype=jnp.int32
            ),
            type_id=jnp.expand_dims(
                jnp.arange(static_params.num_levels), axis=-1
            ).repeat(max_num, axis=-1),
        )

        projectile_directions = jnp.ones(
            (static_params.num_levels, max_num, 2), dtype=jnp.int32
        )

        return projectiles, projectile_directions

    mob_projectiles, mob_projectile_directions = _create_projectiles(
        static_params.max_mob_projectiles
    )
    player_projectiles, player_projectile_directions = _create_projectiles(
        static_params.max_player_projectiles
    )

    # Cows
    passive_mobs = Mobs(
        position=jnp.zeros(
            (static_params.num_levels, static_params.max_passive_mobs, 2),
            dtype=jnp.int32,
        ),
        health=jnp.ones(
            (static_params.num_levels, static_params.max_passive_mobs),
            dtype=jnp.float32,
        )
        * params.passive_mob_health,
        mask=jnp.zeros(
            (static_params.num_levels, static_params.max_passive_mobs), dtype=bool
        ),
        attack_cooldown=jnp.zeros(
            (static_params.num_levels, static_params.max_passive_mobs), dtype=jnp.int32
        ),
        type_id=jnp.expand_dims(jnp.arange(static_params.num_levels), axis=-1).repeat(
            static_params.max_passive_mobs, axis=-1
        ),
    )

    # Plants
    growing_plants_positions = jnp.zeros(
        (static_params.max_growing_plants, 2), dtype=jnp.int32
    )
    growing_plants_age = jnp.zeros(static_params.max_growing_plants, dtype=jnp.int32)
    growing_plants_mask = jnp.zeros(static_params.max_growing_plants, dtype=bool)

    # Potion mapping for episode
    rng, _rng = jax.random.split(rng)
    potion_mapping = jax.random.permutation(_rng, jnp.arange(6))

    rng, _rng = jax.random.split(rng)

    state = EnvState(
        map=map,
        mob_map=jnp.zeros(
            (static_params.num_levels, *static_params.map_size), dtype=bool
        ),
        light_map=light_map,
        down_ladders=ladders_down,
        up_ladders=ladders_up,
        chests_opened=jnp.zeros(static_params.num_levels, dtype=bool),
        monsters_killed=jnp.zeros(static_params.num_levels, dtype=jnp.int32)
        .at[0]
        .set(10),
        player_position=player_position,
        player_direction=Action.UP.value,
        player_level=0,
        player_health=9.0,
        player_food=9,
        player_drink=9,
        player_energy=9,
        player_mana=9,
        player_recover=0.0,
        player_hunger=0.0,
        player_thirst=0.0,
        player_fatigue=0.0,
        player_recover_mana=0.0,
        is_sleeping=False,
        is_resting=False,
        player_xp=0,
        player_dexterity=1,
        player_strength=1,
        player_intelligence=1,
        inventory=Inventory(),
        sword_enchantment=0,
        armour_enchantments=jnp.array([0, 0, 0, 0], dtype=jnp.int32),
        melee_mobs=melee_mobs,
        ranged_mobs=ranged_mobs,
        mob_projectiles=mob_projectiles,
        mob_projectile_directions=mob_projectile_directions,
        player_projectiles=player_projectiles,
        player_projectile_directions=player_projectile_directions,
        passive_mobs=passive_mobs,
        growing_plants_positions=growing_plants_positions,
        growing_plants_age=growing_plants_age,
        growing_plants_mask=growing_plants_mask,
        potion_mapping=potion_mapping,
        learned_spells=jnp.array([False, False], dtype=bool),
        boss_progress=0,
        boss_timesteps_to_spawn_this_round=BOSS_FIGHT_SPAWN_TURNS,
        achievements=jnp.zeros((len(Achievement),), dtype=bool),
        light_level=calculate_light_level(0, params),
        state_rng=_rng,
        timestep=0,
    )

    return state


def generate_random_world(rng, params, static_params):
    # Zombies

    z_pos = jnp.zeros(
        (static_params.num_levels, static_params.max_melee_mobs, 2), dtype=jnp.int32
    )
    z_health = jnp.ones(
        (static_params.num_levels, static_params.max_melee_mobs), dtype=jnp.int32
    )
    z_mask = jnp.zeros(
        (static_params.num_levels, static_params.max_melee_mobs), dtype=bool
    )

    # z_pos = z_pos.at[0, 0].set(player_position + jnp.array([1, 0]))
    # z_mask = z_mask.at[0, 0].set(True)
    # z_pos = z_pos.at[1].set(player_position + jnp.array([2, 0]))
    # z_mask = z_mask.at[1].set(True)

    melee_mobs = Mobs(
        position=z_pos,
        health=z_health,
        mask=z_mask,
        attack_cooldown=jnp.zeros(
            (static_params.num_levels, static_params.max_melee_mobs), dtype=jnp.int32
        ),
        type_id=jnp.expand_dims(jnp.arange(static_params.num_levels), axis=-1).repeat(
            static_params.max_melee_mobs, axis=-1
        ),
    )

    # Skeletons
    sk_positions = jnp.zeros(
        (static_params.num_levels, static_params.max_ranged_mobs, 2), dtype=jnp.int32
    )
    sk_healths = jnp.zeros(
        (static_params.num_levels, static_params.max_ranged_mobs), dtype=jnp.int32
    )
    sk_mask = jnp.zeros(
        (static_params.num_levels, static_params.max_ranged_mobs), dtype=bool
    )

    ranged_mobs = Mobs(
        position=sk_positions,
        health=sk_healths,
        mask=sk_mask,
        attack_cooldown=jnp.zeros(
            (static_params.num_levels, static_params.max_ranged_mobs), dtype=jnp.int32
        ),
        type_id=jnp.expand_dims(jnp.arange(static_params.num_levels), axis=-1).repeat(
            static_params.max_ranged_mobs, axis=-1
        ),
    )

    # Projectiles
    def _create_projectiles(max_num):
        projectile_positions = jnp.zeros(
            (static_params.num_levels, max_num, 2), dtype=jnp.int32
        )
        projectile_healths = jnp.zeros(
            (static_params.num_levels, max_num), dtype=jnp.int32
        )
        projectile_masks = jnp.zeros((static_params.num_levels, max_num), dtype=bool)

        projectiles = Mobs(
            position=projectile_positions,
            health=projectile_healths,
            mask=projectile_masks,
            attack_cooldown=jnp.zeros(
                (static_params.num_levels, max_num), dtype=jnp.int32
            ),
            type_id=jnp.expand_dims(
                jnp.arange(static_params.num_levels), axis=-1
            ).repeat(max_num, axis=-1),
        )

        projectile_directions = jnp.ones(
            (static_params.num_levels, max_num, 2), dtype=jnp.int32
        )

        return projectiles, projectile_directions

    mob_projectiles, mob_projectile_directions = _create_projectiles(
        static_params.max_mob_projectiles
    )
    player_projectiles, player_projectile_directions = _create_projectiles(
        static_params.max_player_projectiles
    )

    # Cows
    passive_mobs = Mobs(
        position=jnp.zeros(
            (static_params.num_levels, static_params.max_passive_mobs, 2),
            dtype=jnp.int32,
        ),
        health=jnp.ones(
            (static_params.num_levels, static_params.max_passive_mobs), dtype=jnp.int32
        )
        * params.passive_mob_health,
        mask=jnp.zeros(
            (static_params.num_levels, static_params.max_passive_mobs), dtype=bool
        ),
        attack_cooldown=jnp.zeros(
            (static_params.num_levels, static_params.max_passive_mobs), dtype=jnp.int32
        ),
        type_id=jnp.expand_dims(jnp.arange(static_params.num_levels), axis=-1).repeat(
            static_params.max_passive_mobs, axis=-1
        ),
    )

    # Plants
    growing_plants_positions = jnp.zeros(
        (static_params.max_growing_plants, 2), dtype=jnp.int32
    )
    growing_plants_age = jnp.zeros(static_params.max_growing_plants, dtype=jnp.int32)
    growing_plants_mask = jnp.zeros(static_params.max_growing_plants, dtype=bool)

    # Potion mapping for episode
    rng, _rng = jax.random.split(rng)
    potion_mapping = jax.random.permutation(_rng, jnp.arange(6))

    rng, _rng = jax.random.split(rng)

    rng, _rng = jax.random.split(rng)
    map = jax.random.choice(
        _rng, jnp.arange(2, 17), shape=(9, *static_params.map_size)
    ).astype(int)

    light_map = jnp.ones_like(map).astype(float)
    ladders_down = jnp.zeros((9, 2), dtype=int)
    ladders_up = jnp.zeros((9, 2), dtype=int)

    state = EnvState(
        map=map,
        mob_map=jnp.zeros(
            (static_params.num_levels, *static_params.map_size), dtype=bool
        ),
        light_map=light_map,
        down_ladders=ladders_down,
        up_ladders=ladders_up,
        chests_opened=jnp.zeros(static_params.num_levels, dtype=bool),
        monsters_killed=jnp.zeros(static_params.num_levels, dtype=jnp.int32)
        .at[0]
        .set(10),
        player_position=jnp.zeros((2,), dtype=int),
        player_direction=Action.UP.value,
        player_level=8,
        player_health=9.0,
        player_food=9,
        player_drink=9,
        player_energy=9,
        player_mana=9,
        player_recover=0.0,
        player_hunger=0.0,
        player_thirst=0.0,
        player_fatigue=0.0,
        player_recover_mana=0.0,
        is_sleeping=False,
        inventory=Inventory(),
        sword_enchantment=0,
        armour_enchantments=jnp.array([0, 1, 2, 1], dtype=jnp.int32),
        melee_mobs=melee_mobs,
        ranged_mobs=ranged_mobs,
        mob_projectiles=mob_projectiles,
        mob_projectile_directions=mob_projectile_directions,
        player_projectiles=player_projectiles,
        player_projectile_directions=player_projectile_directions,
        passive_mobs=passive_mobs,
        growing_plants_positions=growing_plants_positions,
        growing_plants_age=growing_plants_age,
        growing_plants_mask=growing_plants_mask,
        potion_mapping=jnp.zeros((6,), dtype=int),
        learned_spells=jnp.array([False, False], dtype=bool),
        boss_progress=0,
        boss_timesteps_to_spawn_this_round=BOSS_FIGHT_SPAWN_TURNS,
        achievements=jnp.zeros((len(Achievement),), dtype=bool),
        light_level=calculate_light_level(0, params),
        state_rng=_rng,
        timestep=0,
    )

    return state
