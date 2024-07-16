from functools import partial

from craftax.craftax_classic.constants import *
from craftax.craftax_classic.game_logic import calculate_light_level, get_distance_map
from craftax.craftax_classic.envs.craftax_state import EnvState, Inventory, Mobs
from craftax.craftax_classic.util.noise import generate_fractal_noise_2d


def generate_world(rng, params, static_params):
    fractal_noise_angles = params.fractal_noise_angles
    rng, _rng = jax.random.split(rng, num=2)

    player_position = jnp.array(
        [static_params.map_size[0] // 2, static_params.map_size[1] // 2]
    )

    player_proximity_map = get_distance_map(player_position, static_params).astype(
        jnp.float32
    )
    player_proximity_map /= 5.0
    player_proximity_map = jnp.clip(player_proximity_map, 0.0, 1.0)

    larger_res = (static_params.map_size[0] // 4, static_params.map_size[1] // 4)
    large_res = (static_params.map_size[0] // 8, static_params.map_size[1] // 8)
    small_res = (static_params.map_size[0] // 16, static_params.map_size[1] // 16)
    x_res = (static_params.map_size[0] // 8, static_params.map_size[1] // 2)

    # small_res = large_res
    # x_res = large_res

    water = generate_fractal_noise_2d(
        _rng,
        static_params.map_size,
        small_res,
        octaves=1,
        override_angles=fractal_noise_angles[0],
    )
    water = water + player_proximity_map - 1.0

    # Water
    rng, _rng = jax.random.split(rng)
    map = jnp.where(water > 0.7, BlockType.WATER.value, BlockType.GRASS.value)

    # water = water - 0.15 * mountain + 0.15

    # c_water_map = map == BlockType.WATER.value
    # z = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    # c_water_map = jsp.signal.convolve(c_water_map, z, mode="same")

    sand_map = jnp.logical_and(
        water < 0.75,
        jnp.logical_and(
            water > 0.6,
            map != BlockType.WATER.value,
        ),
    )

    # sand_map = jnp.logical_and(
    #     sand_map,
    #     c_water_map > 0.5,
    # )

    map = jnp.where(sand_map, BlockType.SAND.value, map)

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
    mountain = mountain + player_proximity_map - 1.0
    map = jnp.where(mountain > mountain_threshold, BlockType.STONE.value, map)

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
    map = jnp.where(path > 0.5, BlockType.PATH.value, map)

    path_y = path_x.T
    path = jnp.logical_and(mountain > mountain_threshold, path_y > 0.8)
    map = jnp.where(path > 0.5, BlockType.PATH.value, map)

    # Caves
    rng, _rng = jax.random.split(rng)
    caves = jnp.logical_and(mountain > 0.85, water > 0.4)
    map = jnp.where(caves > 0.5, BlockType.PATH.value, map)

    # Ores
    rng, _rng = jax.random.split(rng)
    coal_map = jnp.logical_and(
        map == BlockType.STONE.value,
        jax.random.uniform(_rng, static_params.map_size) < 0.04,
    )
    map = jnp.where(coal_map, BlockType.COAL.value, map)

    rng, _rng = jax.random.split(rng)
    iron_map = jnp.logical_and(
        map == BlockType.STONE.value,
        jax.random.uniform(_rng, static_params.map_size) < 0.03,
    )
    map = jnp.where(iron_map, BlockType.IRON.value, map)

    rng, _rng = jax.random.split(rng)
    diamond_map = jnp.logical_and(
        mountain > 0.8, jax.random.uniform(_rng, static_params.map_size) < 0.005
    )
    diamond_map = jnp.logical_and(diamond_map, map == BlockType.STONE.value)
    map = jnp.where(diamond_map, BlockType.DIAMOND.value, map)

    # Trees
    rng, _rng = jax.random.split(rng)
    tree_noise = generate_fractal_noise_2d(
        _rng,
        static_params.map_size,
        larger_res,
        octaves=1,
        override_angles=fractal_noise_angles[3],
    )
    tree = (tree_noise > 0.5) * jax.random.uniform(
        rng, shape=static_params.map_size
    ) > 0.8
    tree = jnp.logical_and(tree, map == BlockType.GRASS.value)
    map = jnp.where(tree, BlockType.TREE.value, map)

    # Lava
    lava_map = jnp.logical_and(
        mountain > 0.85,
        tree_noise > 0.7,
    )
    map = jnp.where(lava_map, BlockType.LAVA.value, map)

    # Make sure player spawns on grass
    map = map.at[player_position[0], player_position[1]].set(BlockType.GRASS.value)

    # Add diamond if always_diamond flag is set
    valid_diamond = (map.flatten() == BlockType.STONE.value).astype(jnp.float32)
    rng, _rng = jax.random.split(rng)
    diamond_index = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        p=valid_diamond / valid_diamond.sum(),
    )
    diamond_position = jnp.array(
        [
            diamond_index // static_params.map_size[0],
            diamond_index % static_params.map_size[0],
        ]
    )
    diamond_replace_block = jax.lax.select(
        params.always_diamond, BlockType.DIAMOND.value, BlockType.STONE.value
    )
    map = map.at[diamond_position[0], diamond_position[1]].set(diamond_replace_block)

    # Zombies

    z_pos = jnp.zeros((static_params.max_zombies, 2), dtype=jnp.int32)
    z_health = jnp.ones(static_params.max_zombies, dtype=jnp.int32)
    z_mask = jnp.zeros(static_params.max_zombies, dtype=bool)

    # z_pos = z_pos.at[0].set(player_position + jnp.array([1, 0]))
    # z_mask = z_mask.at[0].set(True)
    # z_pos = z_pos.at[1].set(player_position + jnp.array([2, 0]))
    # z_mask = z_mask.at[1].set(True)

    zombies = Mobs(
        position=z_pos,
        health=z_health,
        mask=z_mask,
        attack_cooldown=jnp.zeros(static_params.max_zombies, dtype=jnp.int32),
    )

    # Skeletons
    sk_positions = jnp.zeros((static_params.max_skeletons, 2), dtype=jnp.int32)
    sk_healths = jnp.zeros(static_params.max_skeletons, dtype=jnp.int32)
    sk_mask = jnp.zeros(static_params.max_skeletons, dtype=bool)

    skeletons = Mobs(
        position=sk_positions,
        health=sk_healths,
        mask=sk_mask,
        attack_cooldown=jnp.zeros(static_params.max_skeletons, dtype=jnp.int32),
    )

    # Arrows
    arrow_positions = jnp.zeros((static_params.max_arrows, 2), dtype=jnp.int32)
    arrow_healths = jnp.zeros(static_params.max_arrows, dtype=jnp.int32)
    arrow_masks = jnp.zeros(static_params.max_arrows, dtype=bool)

    arrows = Mobs(
        position=arrow_positions,
        health=arrow_healths,
        mask=arrow_masks,
        attack_cooldown=jnp.zeros(static_params.max_arrows, dtype=jnp.int32),
    )

    arrow_directions = jnp.ones((static_params.max_arrows, 2), dtype=jnp.int32)

    # Cows
    cows = Mobs(
        position=jnp.zeros((static_params.max_cows, 2), dtype=jnp.int32),
        health=jnp.ones(static_params.max_cows, dtype=jnp.int32) * params.cow_health,
        mask=jnp.zeros(static_params.max_cows, dtype=bool),
        attack_cooldown=jnp.zeros(static_params.max_cows, dtype=jnp.int32),
    )

    # Plants
    growing_plants_positions = jnp.zeros(
        (static_params.max_growing_plants, 2), dtype=jnp.int32
    )
    growing_plants_age = jnp.zeros(static_params.max_growing_plants, dtype=jnp.int32)
    growing_plants_mask = jnp.zeros(static_params.max_growing_plants, dtype=bool)

    rng, _rng = jax.random.split(rng)

    state = EnvState(
        map=map,
        mob_map=jnp.zeros(static_params.map_size, dtype=bool),
        player_position=player_position,
        player_direction=Action.UP.value,
        player_health=9,
        player_food=9,
        player_drink=9,
        player_energy=9,
        player_recover=0.0,
        player_hunger=0.0,
        player_thirst=0.0,
        player_fatigue=0.0,
        is_sleeping=False,
        inventory=Inventory(),
        zombies=zombies,
        skeletons=skeletons,
        arrows=arrows,
        arrow_directions=arrow_directions,
        cows=cows,
        growing_plants_positions=growing_plants_positions,
        growing_plants_age=growing_plants_age,
        growing_plants_mask=growing_plants_mask,
        achievements=jnp.zeros((len(Achievement),), dtype=bool),
        light_level=calculate_light_level(0, params),
        state_rng=_rng,
        timestep=0,
    )

    return state


def generate_random_world(rng, params, static_params):
    # Zombies

    z_pos = jnp.zeros((static_params.max_zombies, 2), dtype=jnp.int32)
    z_health = jnp.ones(static_params.max_zombies, dtype=jnp.int32)
    z_mask = jnp.zeros(static_params.max_zombies, dtype=bool)

    # z_pos = z_pos.at[0].set(player_position + jnp.array([1, 0]))
    # z_mask = z_mask.at[0].set(True)

    zombies = Mobs(
        position=z_pos,
        health=z_health,
        mask=z_mask,
        attack_cooldown=jnp.zeros(static_params.max_zombies, dtype=jnp.int32),
    )

    # Skeletons
    sk_positions = jnp.zeros((static_params.max_skeletons, 2), dtype=jnp.int32)
    sk_healths = jnp.zeros(static_params.max_skeletons, dtype=jnp.int32)
    sk_mask = jnp.zeros(static_params.max_skeletons, dtype=bool)

    skeletons = Mobs(
        position=sk_positions,
        health=sk_healths,
        mask=sk_mask,
        attack_cooldown=jnp.zeros(static_params.max_skeletons, dtype=jnp.int32),
    )

    # Arrows
    arrow_positions = jnp.zeros((static_params.max_arrows, 2), dtype=jnp.int32)
    arrow_healths = jnp.zeros(static_params.max_arrows, dtype=jnp.int32)
    arrow_masks = jnp.zeros(static_params.max_arrows, dtype=bool)

    arrows = Mobs(
        position=arrow_positions,
        health=arrow_healths,
        mask=arrow_masks,
        attack_cooldown=jnp.zeros(static_params.max_arrows, dtype=jnp.int32),
    )

    arrow_directions = jnp.ones((static_params.max_arrows, 2), dtype=jnp.int32)

    # Cows
    cows = Mobs(
        position=jnp.zeros((static_params.max_cows, 2), dtype=jnp.int32),
        health=jnp.ones(static_params.max_cows, dtype=jnp.int32) * params.cow_health,
        mask=jnp.zeros(static_params.max_cows, dtype=bool),
        attack_cooldown=jnp.zeros(static_params.max_cows, dtype=jnp.int32),
    )

    # Plants
    growing_plants_positions = jnp.zeros(
        (static_params.max_growing_plants, 2), dtype=jnp.int32
    )
    growing_plants_age = jnp.zeros(static_params.max_growing_plants, dtype=jnp.int32)
    growing_plants_mask = jnp.zeros(static_params.max_growing_plants, dtype=bool)

    rng, _rng = jax.random.split(rng)
    map = jax.random.choice(
        _rng, jnp.arange(2, 17), shape=static_params.map_size
    ).astype(int)

    state = EnvState(
        map=map,
        player_position=jnp.zeros(2, dtype=jnp.int32),
        player_direction=Action.UP.value,
        player_health=9,
        player_food=9,
        player_drink=9,
        player_energy=9,
        player_recover=0.0,
        player_hunger=0.0,
        player_thirst=0.0,
        player_fatigue=0.0,
        is_sleeping=False,
        inventory=Inventory(),
        zombies=zombies,
        skeletons=skeletons,
        arrows=arrows,
        arrow_directions=arrow_directions,
        cows=cows,
        growing_plants_positions=growing_plants_positions,
        growing_plants_age=growing_plants_age,
        growing_plants_mask=growing_plants_mask,
        achievements=jnp.zeros((22,), dtype=bool),
        light_level=calculate_light_level(0),
        timestep=0,
    )

    return state
