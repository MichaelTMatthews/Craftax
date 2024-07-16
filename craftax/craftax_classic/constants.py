import os
import pathlib
from enum import Enum

import jax
import jax.numpy as jnp
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageEnhance
from craftax.environment_base.util import load_compressed_pickle, save_compressed_pickle

# GAME CONSTANTS
OBS_DIM = (7, 9)
MAX_OBS_DIM = max(OBS_DIM)
assert OBS_DIM[0] % 2 == 1 and OBS_DIM[1] % 2 == 1
BLOCK_PIXEL_SIZE_HUMAN = 64
BLOCK_PIXEL_SIZE_IMG = 16
BLOCK_PIXEL_SIZE_AGENT = 7
INVENTORY_OBS_HEIGHT = 2
TEXTURE_CACHE_FILE = os.path.join(
    pathlib.Path(__file__).parent.resolve(), "assets", "texture_cache_classic.pbz2"
)

# ENUMS
class BlockType(Enum):
    INVALID = 0
    OUT_OF_BOUNDS = 1
    GRASS = 2
    WATER = 3
    STONE = 4
    TREE = 5
    WOOD = 6
    PATH = 7
    COAL = 8
    IRON = 9
    DIAMOND = 10
    CRAFTING_TABLE = 11
    FURNACE = 12
    SAND = 13
    LAVA = 14
    PLANT = 15
    RIPE_PLANT = 16


class Action(Enum):
    NOOP = 0  #
    LEFT = 1  # a
    RIGHT = 2  # d
    UP = 3  # w
    DOWN = 4  # s
    DO = 5  # space
    SLEEP = 6  # tab
    PLACE_STONE = 7  # r
    PLACE_TABLE = 8  # t
    PLACE_FURNACE = 9  # f
    PLACE_PLANT = 10  # p
    MAKE_WOOD_PICKAXE = 11  # 1
    MAKE_STONE_PICKAXE = 12  # 2
    MAKE_IRON_PICKAXE = 13  # 3
    MAKE_WOOD_SWORD = 14  # 4
    MAKE_STONE_SWORD = 15  # 5
    MAKE_IRON_SWORD = 16  # 6


# GAME MECHANICS
DIRECTIONS = jnp.concatenate(
    (
        jnp.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=jnp.int32),
        jnp.zeros((11, 2), dtype=jnp.int32),
    ),
    axis=0,
)

CLOSE_BLOCKS = jnp.array(
    [
        [0, -1],
        [0, 1],
        [-1, 0],
        [1, 0],
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
    ],
    dtype=jnp.int32,
)

# Can't walk through these
SOLID_BLOCKS = jnp.array(
    [
        BlockType.WATER.value,
        BlockType.STONE.value,
        BlockType.TREE.value,
        BlockType.COAL.value,
        BlockType.IRON.value,
        BlockType.DIAMOND.value,
        BlockType.CRAFTING_TABLE.value,
        BlockType.FURNACE.value,
        BlockType.PLANT.value,
        BlockType.RIPE_PLANT.value,
    ],
    dtype=jnp.int32,
)


# ACHIEVEMENTS
class Achievement(Enum):
    COLLECT_WOOD = 0
    PLACE_TABLE = 1
    EAT_COW = 2
    COLLECT_SAPLING = 3
    COLLECT_DRINK = 4
    MAKE_WOOD_PICKAXE = 5
    MAKE_WOOD_SWORD = 6
    PLACE_PLANT = 7
    DEFEAT_ZOMBIE = 8
    COLLECT_STONE = 9
    PLACE_STONE = 10
    EAT_PLANT = 11
    DEFEAT_SKELETON = 12
    MAKE_STONE_PICKAXE = 13
    MAKE_STONE_SWORD = 14
    WAKE_UP = 15
    PLACE_FURNACE = 16
    COLLECT_COAL = 17
    COLLECT_IRON = 18
    COLLECT_DIAMOND = 19
    MAKE_IRON_PICKAXE = 20
    MAKE_IRON_SWORD = 21


# TEXTURES
def load_texture(filename, block_pixel_size, clamp_alpha=True):
    filename = os.path.join(pathlib.Path(__file__).parent.resolve(), "assets", filename)
    img = iio.imread(filename)
    jnp_img = jnp.array(img).astype(int)
    assert jnp_img.shape[:2] == (16, 16)

    if jnp_img.shape[2] == 4 and clamp_alpha:
        jnp_img = jnp_img.at[:, :, 3].set(jnp_img[:, :, 3] // 255)

    if block_pixel_size != 16:
        img = np.array(jnp_img, dtype=np.uint8)
        image = Image.fromarray(img)
        image = image.resize(
            (block_pixel_size, block_pixel_size), resample=Image.NEAREST
        )
        jnp_img = jnp.array(image, dtype=jnp.int32)

    return jnp_img


def load_all_textures(block_pixel_size):
    small_block_pixel_size = int(block_pixel_size * 0.8)

    # blocks
    texture_names = [
        "debug_tile.png",
        "debug_tile.png",
        "grass.png",
        "water.png",
        "stone.png",
        "tree.png",
        "wood.png",
        "path.png",
        "coal.png",
        "iron.png",
        "diamond.png",
        "table.png",
        "furnace.png",
        "sand.png",
        "lava.png",
        "plant_on_grass.png",
        "ripe_plant_on_grass.png",
    ]

    block_textures = jnp.array(
        [
            load_texture("debug_tile.png", block_pixel_size),
            jnp.ones((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32) * 128,
            load_texture("grass.png", block_pixel_size),
            load_texture("water.png", block_pixel_size),
            load_texture("stone.png", block_pixel_size),
            load_texture("tree.png", block_pixel_size),
            load_texture("wood.png", block_pixel_size)[:, :, :3],
            load_texture("path.png", block_pixel_size)[:, :, :3],
            load_texture("coal.png", block_pixel_size)[:, :, :3],
            load_texture("iron.png", block_pixel_size)[:, :, :3],
            load_texture("diamond.png", block_pixel_size)[:, :, :3],
            load_texture("table.png", block_pixel_size)[:, :, :3],
            load_texture("furnace.png", block_pixel_size)[:, :, :3],
            load_texture("sand.png", block_pixel_size)[:, :, :3],
            load_texture("lava.png", block_pixel_size)[:, :, :3],
            load_texture("plant_on_grass.png", block_pixel_size)[:, :, :3],
            load_texture("ripe_plant_on_grass.png", block_pixel_size)[:, :, :3],
        ]
    )

    block_textures = jnp.array(
        [load_texture(fname, block_pixel_size)[:, :, :3] for fname in texture_names]
    )
    block_textures = block_textures.at[1].set(
        jnp.ones((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32) * 128
    )

    # rng = jax.random.prngkey(0)
    # block_textures = jax.random.permutation(rng, block_textures)

    smaller_block_textures = jnp.array(
        [
            load_texture(fname, int(block_pixel_size * 0.8))[:, :, :3]
            for fname in texture_names
        ]
    )

    full_map_block_textures = jnp.array(
        [jnp.tile(block_textures[block.value], (*OBS_DIM, 1)) for block in BlockType]
    )

    # player
    pad_pixels = (
        (OBS_DIM[0] // 2) * block_pixel_size,
        (OBS_DIM[1] // 2) * block_pixel_size,
    )

    player_textures = [
        load_texture("player-left.png", block_pixel_size, clamp_alpha=False),
        load_texture("player-right.png", block_pixel_size, clamp_alpha=False),
        load_texture("player-up.png", block_pixel_size, clamp_alpha=False),
        load_texture("player-down.png", block_pixel_size, clamp_alpha=False),
        load_texture("player-sleep.png", block_pixel_size, clamp_alpha=False),
    ]

    full_map_player_textures_rgba = [
        jnp.pad(
            player_texture,
            ((pad_pixels[0], pad_pixels[0]), (pad_pixels[1], pad_pixels[1]), (0, 0)),
        )
        for player_texture in player_textures
    ]

    full_map_player_textures = jnp.array(
        [player_texture[:, :, :3] for player_texture in full_map_player_textures_rgba]
    )

    full_map_player_textures_alpha = jnp.array(
        [
            jnp.repeat(
                jnp.expand_dims(player_texture[:, :, 3], axis=-1).astype(float) / 255,
                repeats=3,
                axis=2,
            )
            for player_texture in full_map_player_textures_rgba
        ]
    )

    # inventory

    empty_texture = jnp.zeros((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32)
    smaller_empty_texture = jnp.zeros(
        (int(block_pixel_size * 0.8), int(block_pixel_size * 0.8), 3), dtype=jnp.int32
    )

    ones_texture = jnp.ones((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32)

    number_size = int(block_pixel_size * 0.6)

    number_textures_rgba = [
        jnp.zeros((number_size, number_size, 3), dtype=jnp.int32),
        load_texture("1.png", number_size),
        load_texture("2.png", number_size),
        load_texture("3.png", number_size),
        load_texture("4.png", number_size),
        load_texture("5.png", number_size),
        load_texture("6.png", number_size),
        load_texture("7.png", number_size),
        load_texture("8.png", number_size),
        load_texture("9.png", number_size),
    ]

    number_textures = jnp.array(
        [
            number_texture[:, :, :3]
            * jnp.repeat(jnp.expand_dims(number_texture[:, :, 3], axis=-1), 3, axis=-1)
            for number_texture in number_textures_rgba
        ]
    )

    number_textures_alpha = jnp.array(
        [
            jnp.repeat(
                jnp.expand_dims(number_texture[:, :, 3], axis=-1), repeats=3, axis=2
            )
            for number_texture in number_textures_rgba
        ]
    )

    health_texture = jnp.array(
        load_texture("health.png", small_block_pixel_size)[:, :, :3]
    )
    hunger_texture = jnp.array(
        load_texture("food.png", small_block_pixel_size)[:, :, :3]
    )
    thirst_texture = jnp.array(
        load_texture("drink.png", small_block_pixel_size)[:, :, :3]
    )
    energy_texture = jnp.array(
        load_texture("energy.png", small_block_pixel_size)[:, :, :3]
    )

    # get rid of the cow ghost
    def apply_alpha(texture):
        return texture[:, :, :3] * jnp.repeat(
            jnp.expand_dims(texture[:, :, 3], axis=-1), 3, axis=-1
        )

    wood_pickaxe_texture = jnp.array(
        load_texture("wood_pickaxe.png", small_block_pixel_size)[:, :, :3]
    )  # no ghosts :)
    stone_pickaxe_texture = jnp.array(
        load_texture("stone_pickaxe.png", small_block_pixel_size)
    )
    stone_pickaxe_texture = apply_alpha(stone_pickaxe_texture)
    iron_pickaxe_texture = jnp.array(
        load_texture("iron_pickaxe.png", small_block_pixel_size)
    )
    iron_pickaxe_texture = apply_alpha(iron_pickaxe_texture)

    wood_sword_texture = jnp.array(
        load_texture("wood_sword.png", small_block_pixel_size)
    )
    wood_sword_texture = apply_alpha(wood_sword_texture)
    stone_sword_texture = jnp.array(
        load_texture("stone_sword.png", small_block_pixel_size)
    )
    stone_sword_texture = apply_alpha(stone_sword_texture)
    iron_sword_texture = jnp.array(
        load_texture("iron_sword.png", small_block_pixel_size)
    )
    iron_sword_texture = apply_alpha(iron_sword_texture)

    sapling_texture = jnp.array(
        load_texture("sapling.png", small_block_pixel_size)[:, :, :3]
    )

    # entities
    zombie_texture_rgba = jnp.array(
        load_texture("zombie.png", block_pixel_size, clamp_alpha=False)
    )
    zombie_texture = zombie_texture_rgba[:, :, :3]
    zombie_texture_alpha = jnp.repeat(
        jnp.expand_dims(zombie_texture_rgba[:, :, 3], axis=-1).astype(float) / 255,
        repeats=3,
        axis=2,
    )

    cow_texture_rgba = jnp.array(
        load_texture("cow.png", block_pixel_size, clamp_alpha=False)
    )
    cow_texture = cow_texture_rgba[:, :, :3]
    cow_texture_alpha = jnp.repeat(
        jnp.expand_dims(cow_texture_rgba[:, :, 3], axis=-1).astype(float) / 255,
        repeats=3,
        axis=2,
    )

    skeleton_texture_rgba = jnp.array(
        load_texture("skeleton.png", block_pixel_size, clamp_alpha=False)
    )
    skeleton_texture = skeleton_texture_rgba[:, :, :3]
    skeleton_texture_alpha = jnp.repeat(
        jnp.expand_dims(skeleton_texture_rgba[:, :, 3], axis=-1).astype(float) / 255,
        repeats=3,
        axis=2,
    )

    arrow_texture_rgba = jnp.array(load_texture("arrow-up.png", block_pixel_size))
    arrow_texture = apply_alpha(arrow_texture_rgba)
    arrow_texture_alpha = jnp.repeat(
        jnp.expand_dims(arrow_texture_rgba[:, :, 3], axis=-1), repeats=3, axis=2
    )

    night_texture = (
        jnp.array([[[0, 16, 64]]])
        .repeat(OBS_DIM[0] * block_pixel_size, axis=0)
        .repeat(OBS_DIM[1] * block_pixel_size, axis=1)
    )

    xs, ys = np.meshgrid(
        np.linspace(-1, 1, OBS_DIM[0] * block_pixel_size),
        np.linspace(-1, 1, OBS_DIM[1] * block_pixel_size),
    )
    night_noise_intensity_texture = (
        1 - np.exp(-0.5 * (xs**2 + ys**2) / (0.5**2)).T
    )

    night_noise_intensity_texture = jnp.expand_dims(
        night_noise_intensity_texture, axis=-1
    ).repeat(3, axis=-1)

    return {
        "block_textures": block_textures,
        "smaller_block_textures": smaller_block_textures,
        "full_map_block_textures": full_map_block_textures,
        "player_textures": player_textures,
        "full_map_player_textures": full_map_player_textures,
        "full_map_player_textures_alpha": full_map_player_textures_alpha,
        "empty_texture": empty_texture,
        "smaller_empty_texture": smaller_empty_texture,
        "ones_texture": ones_texture,
        "number_textures": number_textures,
        "number_textures_alpha": number_textures_alpha,
        "health_texture": health_texture,
        "hunger_texture": hunger_texture,
        "thirst_texture": thirst_texture,
        "energy_texture": energy_texture,
        "wood_pickaxe_texture": wood_pickaxe_texture,
        "stone_pickaxe_texture": stone_pickaxe_texture,
        "iron_pickaxe_texture": iron_pickaxe_texture,
        "wood_sword_texture": wood_sword_texture,
        "stone_sword_texture": stone_sword_texture,
        "iron_sword_texture": iron_sword_texture,
        "sapling_texture": sapling_texture,
        "zombie_texture": zombie_texture,
        "zombie_texture_alpha": zombie_texture_alpha,
        "cow_texture": cow_texture,
        "cow_texture_alpha": cow_texture_alpha,
        "skeleton_texture": skeleton_texture,
        "skeleton_texture_alpha": skeleton_texture_alpha,
        "arrow_texture": arrow_texture,
        "arrow_texture_alpha": arrow_texture_alpha,
        "night_texture": night_texture,
        "night_noise_intensity_texture": night_noise_intensity_texture,
    }


if os.path.exists(TEXTURE_CACHE_FILE) and not os.environ.get(
    "CRAFTAX_RELOAD_TEXTURES", False
):
    TEXTURES = load_compressed_pickle(TEXTURE_CACHE_FILE)
else:
    TEXTURES = {
        BLOCK_PIXEL_SIZE_AGENT: load_all_textures(BLOCK_PIXEL_SIZE_AGENT),
        BLOCK_PIXEL_SIZE_IMG: load_all_textures(BLOCK_PIXEL_SIZE_IMG),
        BLOCK_PIXEL_SIZE_HUMAN: load_all_textures(BLOCK_PIXEL_SIZE_HUMAN),
    }
    save_compressed_pickle(TEXTURE_CACHE_FILE, TEXTURES)
