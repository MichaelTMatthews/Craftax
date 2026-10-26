"""Unicode glyph rendering of a Craftax state.

The local view becomes a small grid with one glyph per tile, followed by a HUD
and a legend of the glyphs on screen. This is handy for text-based agents
(e.g. language models) and for debugging. It works on a single (unbatched)
EnvState and runs outside of jit.
"""

import numpy as np

from craftax.craftax.constants import (
    MONSTERS_KILLED_TO_CLEAR_LEVEL,
    OBS_DIM,
    Achievement,
    BlockType,
    ItemType,
)
from craftax.craftax.util.game_logic_utils import (
    get_max_drink,
    get_max_energy,
    get_max_food,
    get_max_health,
    get_max_mana,
    is_boss_vulnerable,
)

BLOCK_GLYPHS = {
    BlockType.INVALID: ("?", "unknown"),
    BlockType.OUT_OF_BOUNDS: ("█", "map edge"),
    BlockType.GRASS: ("·", "grass"),
    BlockType.WATER: ("≈", "water"),
    BlockType.STONE: ("S", "stone"),
    BlockType.TREE: ("♣", "tree"),
    BlockType.WOOD: ("=", "wood"),
    BlockType.PATH: ("·", "path"),
    BlockType.COAL: ("C", "coal"),
    BlockType.IRON: ("I", "iron"),
    BlockType.DIAMOND: ("D", "diamond"),
    BlockType.CRAFTING_TABLE: ("t", "crafting table"),
    BlockType.FURNACE: ("f", "furnace"),
    BlockType.SAND: ("░", "sand"),
    BlockType.LAVA: ("♨", "lava"),
    BlockType.PLANT: ("+", "sapling"),
    BlockType.RIPE_PLANT: ("*", "ripe plant"),
    BlockType.WALL: ("█", "wall"),
    BlockType.DARKNESS: (" ", "darkness"),
    BlockType.WALL_MOSS: ("█", "mossy wall"),
    BlockType.STALAGMITE: ("^", "stalagmite"),
    BlockType.SAPPHIRE: ("♦", "sapphire"),
    BlockType.RUBY: ("▲", "ruby"),
    BlockType.CHEST: ("$", "chest"),
    BlockType.FOUNTAIN: ("⊙", "fountain"),
    BlockType.FIRE_GRASS: (";", "fire grass"),
    BlockType.ICE_GRASS: (";", "ice grass"),
    BlockType.GRAVEL: (":", "gravel"),
    BlockType.FIRE_TREE: ("♣", "fire tree"),
    BlockType.ICE_SHRUB: ("❄", "ice shrub"),
    BlockType.ENCHANTMENT_TABLE_FIRE: ("Ⓔ", "fire enchantment table"),
    BlockType.ENCHANTMENT_TABLE_ICE: ("Ⓔ", "ice enchantment table"),
    BlockType.NECROMANCER: ("N", "necromancer"),
    BlockType.GRAVE: ("⚰", "grave"),
    BlockType.GRAVE2: ("⚰", "grave"),
    BlockType.GRAVE3: ("⚰", "grave"),
    BlockType.NECROMANCER_VULNERABLE: ("N", "necromancer (vulnerable)"),
}

ITEM_GLYPHS = {
    ItemType.TORCH: ("!", "torch"),
    ItemType.LADDER_DOWN: ("▼", "ladder down"),
    ItemType.LADDER_UP: ("△", "ladder up"),
    ItemType.LADDER_DOWN_BLOCKED: ("▼", "ladder down (locked)"),
}

# Indexed by type_id within each mob class.
MELEE_GLYPHS = [
    ("z", "zombie"),
    ("G", "gnome warrior"),
    ("O", "orc soldier"),
    ("L", "lizard"),
    ("K", "knight"),
    ("T", "troll"),
    ("p", "pigman"),
    ("r", "frost troll"),
]
PASSIVE_GLYPHS = [("c", "cow"), ("b", "bat"), ("s", "snail")]
RANGED_GLYPHS = [
    ("k", "skeleton"),
    ("g", "gnome archer"),
    ("W", "orc mage"),
    ("q", "kobold"),
    ("a", "archer"),
    ("d", "deep thing"),
    ("F", "fire elemental"),
    ("i", "ice elemental"),
]
PROJECTILE_GLYPHS = [
    ("-", "arrow"),
    ("†", "dagger"),
    ("✦", "fireball"),
    ("❅", "iceball"),
    ("-", "arrow"),
    ("●", "slimeball"),
    ("✦", "fireball"),
    ("❅", "iceball"),
]

DIRECTIONS = {1: "left", 2: "right", 3: "up", 4: "down"}
PLAYER_GLYPHS = {1: "←", 2: "→", 3: "↑", 4: "↓"}

FLOOR_NAMES = [
    "Overworld",
    "Dungeon",
    "Gnomish Mines",
    "Sewers",
    "Vaults",
    "Troll Mines",
    "Fire Realm",
    "Ice Realm",
    "Graveyard",
]
MATERIALS = ["No", "Wood", "Stone", "Iron", "Diamond"]
ARMOUR_MATERIALS = ["No", "Iron", "Diamond"]
ARMOUR_SLOTS = ["Helmet", "Chestplate", "Leggings", "Boots"]
ENCHANTMENTS = ["", " (Fire)", " (Ice)"]
POTIONS = ["Red", "Green", "Blue", "Pink", "Cyan", "Yellow"]
RESOURCES = ["Wood", "Stone", "Coal", "Iron", "Diamond", "Sapphire", "Ruby", "Sapling"]


def render_craftax_glyphs(state, legend=True):
    """Render a Craftax EnvState as a Unicode grid, a HUD and a legend."""
    level = int(state.player_level)
    position = np.asarray(state.player_position)
    blocks = _local_view(state.map[level], position, BlockType.OUT_OF_BOUNDS.value)
    items = _local_view(state.item_map[level], position, ItemType.NONE.value)
    light = _local_view(state.light_map[level], position, 0.0)
    boss_vulnerable = bool(is_boss_vulnerable(state))
    ladder_open = int(state.monsters_killed[level]) >= MONSTERS_KILLED_TO_CLEAR_LEVEL

    # Blocks, then items, mobs and darkness on top, then the player.
    cells = [[BLOCK_GLYPHS[BlockType(int(b))] for b in row] for row in blocks]
    if boss_vulnerable:
        for row, col in zip(*np.nonzero(blocks == BlockType.NECROMANCER.value)):
            cells[row][col] = BLOCK_GLYPHS[BlockType.NECROMANCER_VULNERABLE]
    for row, col in zip(*np.nonzero(items != ItemType.NONE.value)):
        item = ItemType(int(items[row, col]))
        if item == ItemType.LADDER_DOWN and not ladder_open:
            item = ItemType.LADDER_DOWN_BLOCKED
        cells[row][col] = ITEM_GLYPHS[item]

    player_projectiles = [(g, "your " + m) for g, m in PROJECTILE_GLYPHS]
    for mobs, glyphs in [
        (state.melee_mobs, MELEE_GLYPHS),
        (state.passive_mobs, PASSIVE_GLYPHS),
        (state.ranged_mobs, RANGED_GLYPHS),
        (state.mob_projectiles, PROJECTILE_GLYPHS),
        (state.player_projectiles, player_projectiles),
    ]:
        mask = np.asarray(mobs.mask[level])
        local = (
            np.asarray(mobs.position[level])[mask] - position + np.array(OBS_DIM) // 2
        )
        for (row, col), type_id in zip(local, np.asarray(mobs.type_id[level])[mask]):
            if 0 <= row < OBS_DIM[0] and 0 <= col < OBS_DIM[1]:
                cells[row][col] = glyphs[type_id]

    for row, col in zip(*np.nonzero(light <= 0.05)):
        if blocks[row, col] != BlockType.OUT_OF_BOUNDS.value:
            cells[row][col] = BLOCK_GLYPHS[BlockType.DARKNESS]

    direction = int(state.player_direction)
    you = f"you, facing {DIRECTIONS[direction]}" if direction in DIRECTIONS else "you"
    cells[OBS_DIM[0] // 2][OBS_DIM[1] // 2] = (PLAYER_GLYPHS.get(direction, "@"), you)

    text = "\n".join("".join(glyph for glyph, _ in row) for row in cells)
    text += "\n\n" + _hud(state, ladder_open, boss_vulnerable)
    if legend:
        text += "\n\nLegend:\n" + _legend(cells)
    return text


def _hud(state, ladder_open, boss_vulnerable):
    level = int(state.player_level)
    inv = state.inventory
    row, col = (int(x) for x in state.player_position)
    facing = DIRECTIONS.get(int(state.player_direction), "none")
    light = float(state.light_level)
    daylight = "Day" if light >= 0.6 else "Dusk" if light >= 0.3 else "Night"

    sword = f"{MATERIALS[int(inv.sword)]} Sword"
    if int(inv.sword) > 0:
        sword += ENCHANTMENTS[int(state.sword_enchantment)]
    bow = "No Bow"
    if int(inv.bow) > 0:
        bow = "Bow" + ENCHANTMENTS[int(state.bow_enchantment)]
    armour = [
        f"{ARMOUR_MATERIALS[int(m)]} {slot}{ENCHANTMENTS[int(e)]}"
        for slot, m, e in zip(ARMOUR_SLOTS, inv.armour, state.armour_enchantments)
        if int(m) > 0
    ]
    kills = int(state.monsters_killed[level])
    ladder = (
        "OPEN" if ladder_open else f"CLOSED {kills}/{MONSTERS_KILLED_TO_CLEAR_LEVEL}"
    )
    achievements = int(np.asarray(state.achievements).sum())

    return "\n".join(
        [
            f"HP: {_num(state.player_health)}/{int(get_max_health(state))}    "
            f"Food: {int(state.player_food)}/{int(get_max_food(state))}    "
            f"Drink: {int(state.player_drink)}/{int(get_max_drink(state))}    "
            f"Energy: {int(state.player_energy)}/{int(get_max_energy(state))}    "
            f"Mana: {int(state.player_mana)}/{int(get_max_mana(state))}",
            f"XP: {int(state.player_xp)}    Str: {int(state.player_strength)}    "
            f"Dex: {int(state.player_dexterity)}    "
            f"Int: {int(state.player_intelligence)}",
            f"Floor: {level} ({FLOOR_NAMES[level]})    Pos: ({row},{col})    "
            f"Facing: {facing}    Light: {daylight}    Step: {int(state.timestep)}",
            "[Inventory] "
            + " ".join(f"{r}:{int(getattr(inv, r.lower()))}" for r in RESOURCES),
            f"[Equipment] {MATERIALS[int(inv.pickaxe)]} Pickaxe | {sword} | {bow} | "
            f"Arrows:{int(inv.arrows)} | Torches:{int(inv.torches)} | "
            f"Books:{int(inv.books)}",
            "[Armour] " + (" | ".join(armour) if armour else "None"),
            "[Potions] "
            + " ".join(f"{p}:{int(n)}" for p, n in zip(POTIONS, inv.potions)),
            f"[Status] Sleeping: {_yes(state.is_sleeping)} | "
            f"Resting: {_yes(state.is_resting)} | "
            f"Fireball: {_yes(state.learned_spells[0])} | "
            f"Iceball: {_yes(state.learned_spells[1])} | Ladder: {ladder} | "
            f"Boss: {'vulnerable' if boss_vulnerable else 'no'} | "
            f"Achievements: {achievements}/{len(Achievement)}",
        ]
    )


def _local_view(grid, center, fill):
    """Slice the OBS_DIM window centred on `center` out of a 2D map."""
    pad = max(OBS_DIM)
    padded = np.pad(np.asarray(grid), pad, constant_values=fill)
    row, col = np.asarray(center) - np.array(OBS_DIM) // 2 + pad
    return padded[row : row + OBS_DIM[0], col : col + OBS_DIM[1]]


def _legend(cells):
    meanings = {}
    for row in cells:
        for glyph, meaning in row:
            if glyph != " " and meaning not in meanings.setdefault(glyph, []):
                meanings[glyph].append(meaning)
    return "\n".join(f"{g} — {' / '.join(m)}" for g, m in sorted(meanings.items()))


def _num(x):
    return f"{float(x):.1f}".rstrip("0").rstrip(".")


def _yes(flag):
    return "yes" if bool(flag) else "no"
