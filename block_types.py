from PIL import Image

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
TABLE = 11


block_translation_dict = {
    GRASS: "GRASS",
    WATER: "WATER",
    STONE: "STONE",
    TREE: "TREE",
    WOOD: "WOOD",
    PATH: "PATH",
    COAL: "COAL",
    IRON: "IRON",
    DIAMOND: "DIAMOND",
    CRAFTING_TABLE: "CRAFTING_TABLE",
    FURNACE: "FURNACE",
    SAND: "SAND",
    LAVA: "LAVA",
    PLANT: "PLANT",
    TABLE: "TABLE",
}

# Load block images
block_images = {key: Image.open(f"craftax/craftax_classic/assets/{name.lower()}.png") for key, name in block_translation_dict.items()}

# Load player images based on direction
player_images = {
    3: Image.open("craftax/craftax_classic/assets/player-up.png"),
    2: Image.open("craftax/craftax_classic/assets/player-right.png"),
    0: Image.open("craftax/craftax_classic/assets/player-down.png"),
    1: Image.open("craftax/craftax_classic/assets/player-left.png"),
}