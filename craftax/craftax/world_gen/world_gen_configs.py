import jax
from flax import struct
import jax.numpy as jnp

from craftax.craftax.constants import BlockType


@struct.dataclass
class SmoothGenConfig:
    default_block: int
    sea_block: int
    coast_block: int
    mountain_block: int
    path_block: int
    inner_mountain_block: int
    ore_requirement_blocks: jnp.ndarray
    ores: jnp.ndarray
    ore_chances: jnp.ndarray
    tree_requirement_block: int
    tree: int
    lava: int
    player_spawn: int
    valid_ladder: int
    ladder_up: int
    ladder_down: int
    player_proximity_map_water_strength: int
    player_proximity_map_water_max: int
    player_proximity_map_mountain_strength: int
    player_proximity_map_mountain_max: int
    default_light: float
    water_threshold: float
    sand_threshold: float
    tree_threshold_uniform: float
    tree_threshold_perlin: float


OVERWORLD_CONFIG = SmoothGenConfig(
    default_block=BlockType.GRASS.value,
    sea_block=BlockType.WATER.value,
    coast_block=BlockType.SAND.value,
    mountain_block=BlockType.STONE.value,
    path_block=BlockType.PATH.value,
    inner_mountain_block=BlockType.PATH.value,
    ore_requirement_blocks=jnp.array([BlockType.STONE.value] * 5),
    ores=jnp.array(
        [
            BlockType.COAL.value,
            BlockType.IRON.value,
            BlockType.DIAMOND.value,
            BlockType.OUT_OF_BOUNDS.value,
            BlockType.OUT_OF_BOUNDS.value,
        ]
    ),
    ore_chances=jnp.array([0.03, 0.02, 0.001, 0.0, 0.0]),
    tree_requirement_block=BlockType.GRASS.value,
    tree=BlockType.TREE.value,
    lava=BlockType.LAVA.value,
    player_spawn=BlockType.GRASS.value,
    valid_ladder=BlockType.PATH.value,
    ladder_up=False,
    ladder_down=True,
    player_proximity_map_water_strength=5,
    player_proximity_map_water_max=1,
    player_proximity_map_mountain_strength=5,
    player_proximity_map_mountain_max=1,
    default_light=1.0,
    water_threshold=0.7,
    sand_threshold=0.6,
    tree_threshold_uniform=0.8,
    tree_threshold_perlin=0.5,
)

GNOMISH_MINES_CONFIG = SmoothGenConfig(
    default_block=BlockType.PATH.value,
    sea_block=BlockType.WATER.value,
    coast_block=BlockType.PATH.value,
    mountain_block=BlockType.STONE.value,
    path_block=BlockType.STONE.value,
    inner_mountain_block=BlockType.STONE.value,
    ore_requirement_blocks=jnp.array([BlockType.STONE.value] * 5),
    ores=jnp.array(
        [
            BlockType.COAL.value,
            BlockType.IRON.value,
            BlockType.DIAMOND.value,
            BlockType.SAPPHIRE.value,
            BlockType.RUBY.value,
        ]
    ),
    ore_chances=jnp.array([0.04, 0.02, 0.005, 0.0025, 0.0025]),
    tree_requirement_block=BlockType.PATH.value,
    tree=BlockType.STALAGMITE.value,
    lava=BlockType.LAVA.value,
    player_spawn=BlockType.PATH.value,
    valid_ladder=BlockType.PATH.value,
    ladder_up=True,
    ladder_down=True,
    player_proximity_map_water_strength=5,
    player_proximity_map_water_max=1,
    player_proximity_map_mountain_strength=17,
    player_proximity_map_mountain_max=1.5,
    default_light=0.0,
    water_threshold=0.7,
    sand_threshold=0.6,
    tree_threshold_uniform=0.8,
    tree_threshold_perlin=0.5,
)

TROLL_MINES_CONFIG = SmoothGenConfig(
    default_block=BlockType.PATH.value,
    sea_block=BlockType.WATER.value,
    coast_block=BlockType.PATH.value,
    mountain_block=BlockType.STONE.value,
    path_block=BlockType.STONE.value,
    inner_mountain_block=BlockType.STONE.value,
    ore_requirement_blocks=jnp.array([BlockType.STONE.value] * 5),
    ores=jnp.array(
        [
            BlockType.COAL.value,
            BlockType.IRON.value,
            BlockType.DIAMOND.value,
            BlockType.SAPPHIRE.value,
            BlockType.RUBY.value,
        ]
    ),
    ore_chances=jnp.array([0.04, 0.03, 0.01, 0.01, 0.01]),
    tree_requirement_block=BlockType.PATH.value,
    tree=BlockType.STALAGMITE.value,
    lava=BlockType.LAVA.value,
    player_spawn=BlockType.PATH.value,
    valid_ladder=BlockType.PATH.value,
    ladder_up=True,
    ladder_down=True,
    player_proximity_map_water_strength=5,
    player_proximity_map_water_max=1,
    player_proximity_map_mountain_strength=17,
    player_proximity_map_mountain_max=1.5,
    default_light=0.0,
    water_threshold=0.7,
    sand_threshold=0.6,
    tree_threshold_uniform=0.8,
    tree_threshold_perlin=0.5,
)

FIRE_LEVEL_CONFIG = SmoothGenConfig(
    default_block=BlockType.FIRE_GRASS.value,
    sea_block=BlockType.LAVA.value,
    coast_block=BlockType.SAND.value,
    mountain_block=BlockType.STONE.value,
    path_block=BlockType.STONE.value,
    inner_mountain_block=BlockType.STONE.value,
    ore_requirement_blocks=jnp.array([BlockType.STONE.value] * 5),
    ores=jnp.array(
        [
            BlockType.COAL.value,
            BlockType.IRON.value,
            BlockType.DIAMOND.value,
            BlockType.SAPPHIRE.value,
            BlockType.RUBY.value,
        ]
    ),
    ore_chances=jnp.array([0.05, 0.0, 0.0, 0.0, 0.025]),
    tree_requirement_block=BlockType.FIRE_GRASS.value,
    tree=BlockType.FIRE_TREE.value,
    lava=BlockType.LAVA.value,
    player_spawn=BlockType.FIRE_GRASS.value,
    valid_ladder=BlockType.FIRE_GRASS.value,
    ladder_up=True,
    ladder_down=True,
    player_proximity_map_water_strength=5,
    player_proximity_map_water_max=1,
    player_proximity_map_mountain_strength=5,
    player_proximity_map_mountain_max=1,
    default_light=1.0,
    water_threshold=0.5,
    sand_threshold=0.6,
    tree_threshold_uniform=0.8,
    tree_threshold_perlin=0.5,
)

ICE_LEVEL_CONFIG = SmoothGenConfig(
    default_block=BlockType.ICE_GRASS.value,
    sea_block=BlockType.WATER.value,
    coast_block=BlockType.ICE_GRASS.value,
    mountain_block=BlockType.STONE.value,
    path_block=BlockType.STONE.value,
    inner_mountain_block=BlockType.STONE.value,
    ore_requirement_blocks=jnp.array([BlockType.STONE.value] * 5),
    ores=jnp.array(
        [
            BlockType.COAL.value,
            BlockType.IRON.value,
            BlockType.DIAMOND.value,
            BlockType.SAPPHIRE.value,
            BlockType.RUBY.value,
        ]
    ),
    ore_chances=jnp.array([0.0, 0.0, 0.005, 0.02, 0.0]),
    tree_requirement_block=BlockType.ICE_GRASS.value,
    tree=BlockType.ICE_SHRUB.value,
    lava=BlockType.WATER.value,
    player_spawn=BlockType.ICE_GRASS.value,
    valid_ladder=BlockType.ICE_GRASS.value,
    ladder_up=True,
    ladder_down=True,
    player_proximity_map_water_strength=5,
    player_proximity_map_water_max=1,
    player_proximity_map_mountain_strength=17,
    player_proximity_map_mountain_max=1.5,
    default_light=0.0,
    water_threshold=0.5,
    sand_threshold=0.6,
    tree_threshold_uniform=0.4,
    tree_threshold_perlin=0.5,
)

BOSS_LEVEL_CONFIG = SmoothGenConfig(
    default_block=BlockType.PATH.value,
    sea_block=BlockType.PATH.value,
    coast_block=BlockType.PATH.value,
    mountain_block=BlockType.WALL.value,
    path_block=BlockType.WALL.value,
    inner_mountain_block=BlockType.WALL.value,
    ore_requirement_blocks=jnp.array(
        [
            BlockType.WALL.value,
            BlockType.GRAVE.value,
            BlockType.GRAVE.value,
            BlockType.WALL.value,
            BlockType.WALL.value,
        ]
    ),
    ores=jnp.array(
        [
            BlockType.WALL_MOSS.value,
            BlockType.GRAVE2.value,
            BlockType.GRAVE3.value,
            BlockType.SAPPHIRE.value,
            BlockType.RUBY.value,
        ]
    ),
    ore_chances=jnp.array([0.1, 0.333, 0.5, 0.0, 0.0]),
    tree_requirement_block=BlockType.PATH.value,
    tree=BlockType.GRAVE.value,
    lava=BlockType.WALL.value,
    player_spawn=BlockType.NECROMANCER.value,
    valid_ladder=BlockType.PATH.value,
    ladder_up=False,
    ladder_down=False,
    player_proximity_map_water_strength=5,
    player_proximity_map_water_max=1,
    player_proximity_map_mountain_strength=10,
    player_proximity_map_mountain_max=10,
    default_light=0.0,
    water_threshold=0.7,
    sand_threshold=0.6,
    tree_threshold_uniform=0.95,
    tree_threshold_perlin=-1.0,
)

ALL_SMOOTHGEN_CONFIGS = jax.tree_map(
    lambda l1, l2, l3, l4, l5, l6: jnp.stack((l1, l2, l3, l4, l5, l6), axis=0),
    OVERWORLD_CONFIG,
    GNOMISH_MINES_CONFIG,
    TROLL_MINES_CONFIG,
    FIRE_LEVEL_CONFIG,
    ICE_LEVEL_CONFIG,
    BOSS_LEVEL_CONFIG,
)


@struct.dataclass
class DungeonConfig:
    special_block: int
    fountain_block: int
    rare_path_replacement_block: int


DUNGEON_CONFIG = DungeonConfig(
    special_block=BlockType.PATH.value,
    fountain_block=BlockType.FOUNTAIN.value,
    rare_path_replacement_block=BlockType.PATH.value,
)

SEWER_CONFIG = DungeonConfig(
    special_block=BlockType.ENCHANTMENT_TABLE_ICE.value,
    fountain_block=BlockType.WATER.value,
    rare_path_replacement_block=BlockType.WATER.value,
)

VAULTS_CONFIG = DungeonConfig(
    special_block=BlockType.ENCHANTMENT_TABLE_FIRE.value,
    fountain_block=BlockType.FOUNTAIN.value,
    rare_path_replacement_block=BlockType.PATH.value,
)

ALL_DUNGEON_CONFIGS = jax.tree_map(
    lambda x, y, z: jnp.stack((x, y, z), axis=0),
    DUNGEON_CONFIG,
    SEWER_CONFIG,
    VAULTS_CONFIG,
)
