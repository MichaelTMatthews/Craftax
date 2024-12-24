import jax
import jax.scipy as jsp

from craftax_marl.constants import *
from craftax_marl.game_logic import calculate_light_level
from craftax_marl.util.maths_utils import get_all_players_distance_map
from craftax_marl.craftax_state import EnvState, Inventory, Mobs
from craftax_marl.util.game_logic_utils import get_ladder_positions
from craftax_marl.util.noise import generate_fractal_noise_2d
from craftax_marl.world_gen.world_gen_configs import (
    ALL_DUNGEON_CONFIGS,
    ALL_SMOOTHGEN_CONFIGS,
)


def get_new_empty_inventory(player_count):
    return Inventory(
        wood=jnp.full((player_count,), 0, dtype=jnp.int32),
        stone=jnp.full((player_count,), 0, dtype=jnp.int32),
        coal=jnp.full((player_count,), 0, dtype=jnp.int32),
        iron=jnp.full((player_count,), 0, dtype=jnp.int32),
        diamond=jnp.full((player_count,), 0, dtype=jnp.int32),
        sapling=jnp.full((player_count,), 0, dtype=jnp.int32),
        pickaxe=jnp.full((player_count,), 0, dtype=jnp.int32),
        sword=jnp.full((player_count,), 0, dtype=jnp.int32),
        bow=jnp.full((player_count,), 0, dtype=jnp.int32),
        arrows=jnp.full((player_count,), 0, dtype=jnp.int32),
        torches=jnp.full((player_count,), 0, dtype=jnp.int32),
        ruby=jnp.full((player_count,), 0, dtype=jnp.int32),
        sapphire=jnp.full((player_count,), 0, dtype=jnp.int32),
        books=jnp.full((player_count,), 0, dtype=jnp.int32),
        potions=jnp.full((player_count, 6), 0, dtype=jnp.int32),
        armour=jnp.full((player_count, 4), 0, dtype=jnp.int32),
    )


def get_new_full_inventory(player_count):
    return Inventory(
        wood=jnp.full((player_count,), 99, dtype=jnp.int32),
        stone=jnp.full((player_count,), 99, dtype=jnp.int32),
        coal=jnp.full((player_count,), 99, dtype=jnp.int32),
        iron=jnp.full((player_count,), 99, dtype=jnp.int32),
        diamond=jnp.full((player_count,), 99, dtype=jnp.int32),
        sapling=jnp.full((player_count,), 99, dtype=jnp.int32),
        pickaxe=jnp.full((player_count,), 4, dtype=jnp.int32),
        sword=jnp.full((player_count,), 4, dtype=jnp.int32),
        bow=jnp.full((player_count,), 1, dtype=jnp.int32),
        arrows=jnp.full((player_count,), 99, dtype=jnp.int32),
        torches=jnp.full((player_count,), 99, dtype=jnp.int32),
        ruby=jnp.full((player_count,), 99, dtype=jnp.int32),
        sapphire=jnp.full((player_count,), 99, dtype=jnp.int32),
        books=jnp.full((player_count,), 99, dtype=jnp.int32),
        potions=jnp.full((player_count, 6), 99, dtype=jnp.int32),
        armour=jnp.full((player_count, 4), 2, dtype=jnp.int32),
    )


def generate_dungeon(rng, static_params, config):
    chunk_size = 16
    world_chunk_width = static_params.map_size[0] // chunk_size
    world_chunk_height = static_params.map_size[1] // chunk_size
    room_occupancy_chunks = jnp.ones(world_chunk_width * world_chunk_height)

    rng, _rng, __rng = jax.random.split(rng, 3)
    room_sizes = jax.random.randint(
        __rng, shape=(NUM_ROOMS, 2), minval=MIN_ROOM_SIZE, maxval=MAX_ROOM_SIZE
    )

    map = jnp.ones(static_params.map_size, dtype=jnp.int32) * BlockType.WALL.value
    padded_map = jnp.pad(map, MAX_ROOM_SIZE, constant_values=0)

    item_map = jnp.zeros(static_params.map_size, dtype=jnp.int32)
    padded_item_map = jnp.pad(item_map, MAX_ROOM_SIZE, constant_values=0)

    def _add_room(carry, room_index):
        block_map, item_map, room_occupancy_chunks, rng = carry

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
        ) + jnp.array([MAX_ROOM_SIZE, MAX_ROOM_SIZE])
        rng, _rng = jax.random.split(rng)
        room_position += jax.random.randint(
            _rng, (2,), minval=0, maxval=chunk_size - MIN_ROOM_SIZE
        )

        slice = jax.lax.dynamic_slice(
            block_map, room_position, (MAX_ROOM_SIZE, MAX_ROOM_SIZE)
        )
        xs = jnp.expand_dims(jnp.arange(MAX_ROOM_SIZE), axis=-1).repeat(
            MAX_ROOM_SIZE, axis=-1
        )
        ys = jnp.expand_dims(jnp.arange(MAX_ROOM_SIZE), axis=0).repeat(
            MAX_ROOM_SIZE, axis=0
        )

        room_mask = jnp.logical_and(
            xs < room_sizes[room_index, 0], ys < room_sizes[room_index, 1]
        )

        slice = room_mask * BlockType.PATH.value + (1 - room_mask) * slice

        block_map = jax.lax.dynamic_update_slice(
            block_map,
            slice,
            room_position,
        )

        # Torches in corner
        item_map = item_map.at[room_position[0], room_position[1]].set(
            ItemType.TORCH.value
        )
        item_map = item_map.at[
            room_position[0] + room_sizes[room_index, 0] - 1, room_position[1]
        ].set(ItemType.TORCH.value)
        item_map = item_map.at[
            room_position[0], room_position[1] + room_sizes[room_index, 1] - 1
        ].set(ItemType.TORCH.value)
        item_map = item_map.at[
            room_position[0] + room_sizes[room_index, 0] - 1,
            room_position[1] + room_sizes[room_index, 1] - 1,
        ].set(ItemType.TORCH.value)

        # Chest
        rng, _rng = jax.random.split(rng)
        chest_position = jax.random.randint(
            _rng,
            shape=(static_params.player_count, 2),
            minval=jnp.ones(2),
            maxval=room_sizes[room_index] - jnp.ones(2),
        )
        block_map = block_map.at[
            room_position[0] + chest_position[:, 0], room_position[1] + chest_position[:, 1]
        ].set(BlockType.CHEST.value)

        # Fountain
        rng, _rng, __rng = jax.random.split(rng, 3)
        fountain_position = jax.random.randint(
            _rng,
            shape=(2,),
            minval=jnp.ones(2),
            maxval=room_sizes[room_index] - jnp.ones(2),
        )
        room_has_fountain = jax.random.uniform(__rng) > 0.5
        fountain_block = (
            room_has_fountain * config.fountain_block
            + (1 - room_has_fountain)
            * block_map[
                room_position[0] + fountain_position[0],
                room_position[1] + fountain_position[1],
            ]
        )
        block_map = block_map.at[
            room_position[0] + fountain_position[0],
            room_position[1] + fountain_position[1],
        ].set(fountain_block)

        return (block_map, item_map, room_occupancy_chunks, rng), (room_position, room_position+chest_position-MAX_ROOM_SIZE)

    rng, _rng = jax.random.split(rng)
    (padded_map, padded_item_map, _, _), (room_positions, chest_positions) = jax.lax.scan(
        _add_room,
        (padded_map, padded_item_map, room_occupancy_chunks, _rng),
        jnp.arange(NUM_ROOMS),
    )

    # convert from (rooms, players, 2) to (players, rooms, 2)
    chest_positions = chest_positions.transpose((1, 0, 2))

    def _add_path(carry, path_index):
        cmap, included_rooms_mask, rng = carry

        path_source = room_positions[path_index]

        rng, _rng = jax.random.split(rng)
        sink_index = jax.random.choice(
            _rng, jnp.arange(NUM_ROOMS), p=included_rooms_mask
        )
        path_sink = room_positions[sink_index]

        # Horizontal component
        entire_row = cmap[path_source[0]]
        path_indexes = jnp.arange(static_params.map_size[0] + 2 * MAX_ROOM_SIZE)
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
        path_indexes = jnp.arange(static_params.map_size[1] + 2 * MAX_ROOM_SIZE)
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
    included_rooms_mask = jnp.zeros(NUM_ROOMS, dtype=bool).at[-1].set(True)
    (
        (padded_map, _, _),
        _,
    ) = jax.lax.scan(
        _add_path, (padded_map, included_rooms_mask, _rng), jnp.arange(0, NUM_ROOMS)
    )

    # Place special block in a random room
    special_block_position = room_positions[0] + jnp.array([2, 2])
    padded_map = padded_map.at[
        special_block_position[0], special_block_position[1]
    ].set(config.special_block)

    map = padded_map[MAX_ROOM_SIZE:-MAX_ROOM_SIZE, MAX_ROOM_SIZE:-MAX_ROOM_SIZE]
    item_map = padded_item_map[
        MAX_ROOM_SIZE:-MAX_ROOM_SIZE, MAX_ROOM_SIZE:-MAX_ROOM_SIZE
    ]

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
    rare_map = jnp.logical_and(rare_map, item_map == ItemType.NONE.value)
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
    rng, _rng = jax.random.split(rng)
    ladders_down = get_ladder_positions(_rng, static_params, config, map)
    item_map = item_map.at[ladders_down[:, 0], ladders_down[:, 1]].set(
        ItemType.LADDER_DOWN.value
    )

    rng, _rng = jax.random.split(rng)
    ladders_up = get_ladder_positions(_rng, static_params, config, map)
    item_map = item_map.at[ladders_up[:, 0], ladders_up[:, 1]].set(
        ItemType.LADDER_UP.value
    )

    return map, item_map, light_map, ladders_down, ladders_up, chest_positions


def generate_smoothworld(rng, static_params, player_position, config, params=None):
    if params is not None:
        fractal_noise_angles = params.fractal_noise_angles
    else:
        fractal_noise_angles = (None, None, None, None, None)

    player_proximity_map = get_all_players_distance_map(
        player_position, jnp.full(static_params.player_count, True), static_params
    )
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
        _rng,
        static_params.map_size,
        small_res,
        octaves=1,
        override_angles=fractal_noise_angles[0],
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
        generate_fractal_noise_2d(
            _rng,
            static_params.map_size,
            small_res,
            octaves=1,
            override_angles=fractal_noise_angles[1],
        )
        + 0.05
    )
    mountain = mountain + player_proximity_map_mountain - 1.0
    map = jnp.where(mountain > mountain_threshold, config.mountain_block, map)

    # Paths
    rng, _rng = jax.random.split(rng)
    path_x = generate_fractal_noise_2d(
        _rng,
        static_params.map_size,
        x_res,
        octaves=1,
        override_angles=fractal_noise_angles[2],
    )
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
        _rng,
        static_params.map_size,
        larger_res,
        octaves=1,
        override_angles=fractal_noise_angles[3],
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

    # Light map
    light_map = (
        jnp.ones(static_params.map_size, dtype=jnp.float32) * config.default_light
    )

    # Make sure player spawns on grass
    map = map.at[player_position[:, 0], player_position[:, 1]].set(config.player_spawn)

    item_map = jnp.zeros(static_params.map_size, dtype=jnp.int32)

    rng, _rng = jax.random.split(rng)
    ladders_down = get_ladder_positions(_rng, static_params, config, map)

    item_map = item_map.at[ladders_down[:, 0], ladders_down[:, 1]].set(
        ItemType.LADDER_DOWN.value * config.ladder_down
        + map[ladders_down[:, 0], ladders_down[:, 1]] * (1 - config.ladder_down)
    )

    rng, _rng = jax.random.split(rng)
    ladders_up = get_ladder_positions(_rng, static_params, config, map)

    LIGHT_MAP_AROUND_LADDER = TORCH_LIGHT_MAP * (
        1 - config.default_light
    ) + config.default_light * jnp.ones((9, 9))

    def _set_ladder_light(light_map, ladder_position):
        out = jax.lax.dynamic_update_slice(
            light_map, LIGHT_MAP_AROUND_LADDER, ladder_position - jnp.array([4, 4])
        )
        return out, None

    light_map, _ = jax.lax.scan(_set_ladder_light, light_map, ladders_up)

    z = jnp.array([[0.2, 0.7, 0.2], [0.7, 1, 0.7], [0.2, 0.7, 0.2]]) * (
        config.lava == BlockType.LAVA.value
    )
    light_map += jsp.signal.convolve(lava_map, z, mode="same")
    light_map = jnp.clip(light_map, 0.0, 1.0)

    item_map = item_map.at[ladders_up[:, 0], ladders_up[:, 1]].set(
        ItemType.LADDER_UP.value * config.ladder_up
        + map[ladders_up[:, 0], ladders_up[:, 1]] * (1 - config.ladder_up)
    )

    # No chests exist in smoothworlds and so we return an empty array here
    chest_positions = jnp.zeros((static_params.player_count, NUM_ROOMS, 2), dtype=jnp.int32)

    return map, item_map, light_map, ladders_down, ladders_up, chest_positions


def generate_world(rng, params, static_params):
    # Start players in the middle of the map
    def get_player_spawn(idx):
        width = jnp.ceil(jnp.sqrt(static_params.player_count)).astype(jnp.int32)
        return jnp.array(
            [
                (static_params.map_size[0] // 2) + (idx // width),
                (static_params.map_size[1] // 2) + (idx % width),
            ]
        )

    player_position = jax.vmap(get_player_spawn)(
        jnp.arange(0, static_params.player_count)
    )

    # Generate smoothgens (overworld, caves, elemental levels, boss level)
    rngs = jax.random.split(rng, 7)
    rng, _rng = rngs[0], rngs[1:]
    smoothgens = jax.vmap(generate_smoothworld, in_axes=(0, None, None, 0))(
        _rng, static_params, player_position, ALL_SMOOTHGEN_CONFIGS
    )

    # Generate dungeons
    rngs = jax.random.split(rng, 4)
    rng, _rng = rngs[0], rngs[1:]
    dungeons = jax.vmap(generate_dungeon, in_axes=(0, None, 0))(
        _rng, static_params, ALL_DUNGEON_CONFIGS
    )

    # Returns stacked versions of the map, item_map, light_map and ladders
    # 9 elements in each of these stacks representing each of the levels.
    # Splice smoothgens and dungeons in order of levels
    map, item_map, light_map, ladders_down, ladders_up, chest_positions = jax.tree_map(
        lambda x, y: jnp.stack(
            (x[0], y[0], x[1], y[1], y[2], x[2], x[3], x[4], x[5]), axis=0
        ),
        smoothgens,
        dungeons,
    )

    # Mobs
    def generate_empty_mobs(max_mobs):
        return Mobs(
            position=jnp.zeros(
                (static_params.num_levels, max_mobs, 2), dtype=jnp.int32
            ),
            health=jnp.ones((static_params.num_levels, max_mobs), dtype=jnp.float32),
            mask=jnp.zeros((static_params.num_levels, max_mobs), dtype=bool),
            attack_cooldown=jnp.zeros(
                (static_params.num_levels, max_mobs), dtype=jnp.int32
            ),
            type_id=jnp.zeros((static_params.num_levels, max_mobs), dtype=jnp.int32),
        )

    melee_mobs = generate_empty_mobs(
        static_params.max_melee_mobs * static_params.player_count
    )
    ranged_mobs = generate_empty_mobs(
        static_params.max_ranged_mobs * static_params.player_count
    )
    passive_mobs = generate_empty_mobs(
        static_params.max_passive_mobs * static_params.player_count
    )

    # Projectiles
    def _create_projectiles(max_num):
        projectiles = generate_empty_mobs(max_num)

        projectile_directions = jnp.ones(
            (static_params.num_levels, max_num, 2), dtype=jnp.int32
        )

        projectile_owners = jnp.zeros(
            (static_params.num_levels, max_num), dtype=jnp.int32
        )

        return projectiles, projectile_directions, projectile_owners

    mob_projectiles, mob_projectile_directions, mob_projectile_owners = _create_projectiles(
        static_params.max_mob_projectiles * static_params.player_count
    )
    player_projectiles, player_projectile_directions, player_projectile_owners = _create_projectiles(
        static_params.max_player_projectiles * static_params.player_count
    )

    # Plants
    growing_plants_positions = jnp.zeros(
        (static_params.max_growing_plants * static_params.player_count, 2), dtype=jnp.int32
    )
    growing_plants_age = jnp.zeros(static_params.max_growing_plants * static_params.player_count, dtype=jnp.int32)
    growing_plants_mask = jnp.zeros(static_params.max_growing_plants * static_params.player_count, dtype=bool)

    # Potion mapping for episode
    rng, _rng = jax.random.split(rng)
    potion_mapping = jax.random.permutation(_rng, jnp.arange(6))

    # Inventory
    inventory = jax.tree_map(
        lambda x, y: jax.lax.select(params.god_mode, x, y),
        get_new_full_inventory(static_params.player_count),
        get_new_empty_inventory(static_params.player_count),
    )

    rng, _rng = jax.random.split(rng)

    state = EnvState(
        map=map,
        item_map=item_map,
        mob_map=jnp.zeros(
            (static_params.num_levels, *static_params.map_size), dtype=bool
        ),
        light_map=light_map,
        down_ladders=ladders_down,
        up_ladders=ladders_up,
        chests_opened=jnp.zeros((static_params.num_levels, static_params.player_count), dtype=bool),
        chest_positions=chest_positions,
        monsters_killed=jnp.zeros(static_params.num_levels, dtype=jnp.int32)
        .at[0]
        .set(10),  # First ladder starts open
        player_position=player_position,
        player_direction=jnp.full(
            (static_params.player_count,), Action.UP.value, dtype=jnp.int32
        ),
        player_level=jnp.asarray(0, dtype=jnp.int32),
        player_health=jnp.full((static_params.player_count,), 9.0, dtype=jnp.float32),
        player_alive=jnp.full((static_params.player_count,), True, dtype=bool),
        player_food=jnp.full((static_params.player_count,), 9, dtype=jnp.int32),
        player_drink=jnp.full((static_params.player_count,), 9, dtype=jnp.int32),
        player_energy=jnp.full((static_params.player_count,), 9, dtype=jnp.int32),
        player_mana=jnp.full((static_params.player_count,), 9, dtype=jnp.int32),
        player_recover=jnp.full((static_params.player_count,), 0.0, dtype=jnp.float32),
        player_hunger=jnp.full((static_params.player_count,), 0.0, dtype=jnp.float32),
        player_thirst=jnp.full((static_params.player_count,), 0.0, dtype=jnp.float32),
        player_fatigue=jnp.full((static_params.player_count,), 0.0, dtype=jnp.float32),
        player_recover_mana=jnp.full(
            (static_params.player_count,), 0.0, dtype=jnp.float32
        ),
        is_sleeping=jnp.full((static_params.player_count,), False, dtype=jnp.bool),
        is_resting=jnp.full((static_params.player_count,), False, dtype=jnp.bool),
        player_xp=jnp.full((static_params.player_count,), 0, dtype=jnp.int32),
        player_dexterity=jnp.full((static_params.player_count,), 1, dtype=jnp.int32),
        player_strength=jnp.full((static_params.player_count,), 1, dtype=jnp.int32),
        player_intelligence=jnp.full((static_params.player_count,), 1, dtype=jnp.int32),
        request_duration=jnp.full((static_params.player_count,), 0, dtype=jnp.int32),
        request_type=jnp.full((static_params.player_count,), 0, dtype=jnp.int32),
        inventory=inventory,
        sword_enchantment=jnp.full((static_params.player_count,), 0, dtype=jnp.int32),
        bow_enchantment=jnp.full((static_params.player_count,), 0, dtype=jnp.int32),
        armour_enchantments=jnp.full(
            (static_params.player_count, 4), 0, dtype=jnp.int32
        ),
        melee_mobs=melee_mobs,
        ranged_mobs=ranged_mobs,
        passive_mobs=passive_mobs,
        mob_projectiles=mob_projectiles,
        mob_projectile_directions=mob_projectile_directions,
        mob_projectile_owners=mob_projectile_owners,
        player_projectiles=player_projectiles,
        player_projectile_directions=player_projectile_directions,
        player_projectile_owners=player_projectile_owners,
        growing_plants_positions=growing_plants_positions,
        growing_plants_age=growing_plants_age,
        growing_plants_mask=growing_plants_mask,
        potion_mapping=potion_mapping,
        learned_spells=jnp.full((static_params.player_count, 2), False, dtype=jnp.bool),
        boss_progress=jnp.asarray(0, dtype=jnp.int32),
        boss_timesteps_to_spawn_this_round=jnp.asarray(
            BOSS_FIGHT_SPAWN_TURNS, dtype=jnp.int32
        ),
        achievements=jnp.zeros(
            (static_params.player_count, len(Achievement)), dtype=bool
        ),
        light_level=jnp.asarray(calculate_light_level(0, params), dtype=jnp.float32),
        state_rng=_rng,
        timestep=jnp.asarray(0, dtype=jnp.int32),
    )

    return state
