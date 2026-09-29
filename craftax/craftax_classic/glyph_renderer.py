"""Unicode glyph rendering of a Craftax-Classic state.

The local view becomes a small grid with one glyph per tile, followed by a HUD
and a legend of the glyphs on screen. This is handy for text-based agents
(e.g. language models) and for debugging. It works on a single (unbatched)
EnvState and runs outside of jit.
"""

import numpy as np

from craftax.craftax_classic.constants import OBS_DIM, Achievement, BlockType

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
}

DIRECTIONS = {1: "left", 2: "right", 3: "up", 4: "down"}
PLAYER_GLYPHS = {1: "←", 2: "→", 3: "↑", 4: "↓"}
RESOURCES = ["Wood", "Stone", "Coal", "Iron", "Diamond", "Sapling"]
TOOLS = ["wood_pickaxe", "stone_pickaxe", "iron_pickaxe"]
TOOLS += ["wood_sword", "stone_sword", "iron_sword"]


def render_craftax_glyphs(state, legend=True):
    """Render a Craftax-Classic EnvState as a Unicode grid, a HUD and a legend."""
    position = np.asarray(state.player_position)
    blocks = _local_view(state.map, position, BlockType.OUT_OF_BOUNDS.value)

    # Blocks, then mobs on top, then the player.
    cells = [[BLOCK_GLYPHS[BlockType(int(b))] for b in row] for row in blocks]
    for mobs, glyph in [
        (state.zombies, ("z", "zombie")),
        (state.cows, ("c", "cow")),
        (state.skeletons, ("k", "skeleton")),
        (state.arrows, ("-", "arrow")),
    ]:
        mask = np.asarray(mobs.mask)
        local = np.asarray(mobs.position)[mask] - position + np.array(OBS_DIM) // 2
        for row, col in local:
            if 0 <= row < OBS_DIM[0] and 0 <= col < OBS_DIM[1]:
                cells[row][col] = glyph

    direction = int(state.player_direction)
    you = f"you, facing {DIRECTIONS[direction]}" if direction in DIRECTIONS else "you"
    cells[OBS_DIM[0] // 2][OBS_DIM[1] // 2] = (PLAYER_GLYPHS.get(direction, "@"), you)

    text = "\n".join("".join(glyph for glyph, _ in row) for row in cells)
    text += "\n\n" + _hud(state)
    if legend:
        text += "\n\nLegend:\n" + _legend(cells)
    return text


def _hud(state):
    inv = state.inventory
    row, col = (int(x) for x in state.player_position)
    facing = DIRECTIONS.get(int(state.player_direction), "none")
    light = float(state.light_level)
    daylight = "Day" if light >= 0.6 else "Dusk" if light >= 0.3 else "Night"
    achievements = int(np.asarray(state.achievements).sum())
    tools = [f"{t.replace('_', ' ').title()}:{int(getattr(inv, t))}" for t in TOOLS]

    return "\n".join(
        [
            f"HP: {int(state.player_health)}/9    Food: {int(state.player_food)}/9    "
            f"Drink: {int(state.player_drink)}/9    "
            f"Energy: {int(state.player_energy)}/9",
            f"Pos: ({row},{col})    Facing: {facing}    Light: {daylight}    "
            f"Step: {int(state.timestep)}",
            "[Inventory] "
            + " ".join(f"{r}:{int(getattr(inv, r.lower()))}" for r in RESOURCES),
            "[Tools] " + " ".join(tools),
            f"[Status] Sleeping: {'yes' if state.is_sleeping else 'no'} | "
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
