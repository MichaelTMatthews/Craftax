from craftax.craftax.util.game_logic_utils import *


def is_game_over(state, params, static_env_params):
    done_steps = state.timestep >= params.max_timesteps
    is_dead = state.player_health <= 0
    defeated_boss = has_beaten_boss(state, static_env_params)

    return done_steps | is_dead | defeated_boss


def update_plants_with_eat(state, plant_position, static_params):
    def _is_plant(unused, index):
        return None, (state.growing_plants_positions[index] == plant_position).all()

    _, is_plant = jax.lax.scan(
        _is_plant, None, jnp.arange(static_params.max_growing_plants)
    )

    plant_index = jnp.argmax(is_plant)

    return state.growing_plants_age.at[plant_index].set(0)


def add_items_from_chest(rng, state, inventory, is_opening_chest):
    # Wood (60%)
    rng, _rng = jax.random.split(rng)
    is_looting_wood = jax.random.uniform(_rng) < 0.6
    rng, _rng = jax.random.split(rng)
    wood_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=1, maxval=6) * is_looting_wood
    )

    # Torch (60%)
    rng, _rng = jax.random.split(rng)
    is_looting_torch = jax.random.uniform(_rng) < 0.6
    rng, _rng = jax.random.split(rng)
    torch_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=4, maxval=8) * is_looting_torch
    )

    # Ores (60%)
    rng, _rng = jax.random.split(rng)
    is_looting_ore = jax.random.uniform(_rng) < 0.6
    rng, _rng = jax.random.split(rng)
    ore_loot_id = jax.random.choice(
        _rng,
        jnp.arange(5, dtype=jnp.int32),
        shape=(),
        p=jnp.array([0.3, 0.3, 0.15, 0.125, 0.125]),
    )
    rng, _rng = jax.random.split(rng)

    # Use the same rng as events are mutually exclusive
    coal_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=1, maxval=4)
        * (ore_loot_id == 0)
        * is_looting_ore
    )
    iron_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=1, maxval=3)
        * (ore_loot_id == 1)
        * is_looting_ore
    )
    diamond_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=1, maxval=2)
        * (ore_loot_id == 2)
        * is_looting_ore
    )
    sapphire_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=1, maxval=2)
        * (ore_loot_id == 3)
        * is_looting_ore
    )
    ruby_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=1, maxval=2)
        * (ore_loot_id == 4)
        * is_looting_ore
    )

    # Potion (50%)
    rng, _rng = jax.random.split(rng)
    is_looting_potion = jax.random.uniform(_rng) < 0.5
    rng, _rng = jax.random.split(rng)
    potion_loot_index = jax.random.randint(_rng, shape=(), minval=0, maxval=6)
    rng, _rng = jax.random.split(rng)
    potion_loot_amount = jax.random.randint(_rng, shape=(), minval=1, maxval=3)

    # Arrows (25%)
    rng, _rng = jax.random.split(rng)
    is_looting_arrows = jax.random.uniform(_rng) < 0.25
    rng, _rng = jax.random.split(rng)
    arrows_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=1, maxval=5) * is_looting_arrows
    )

    # Tools (20%)
    rng, _rng = jax.random.split(rng)
    is_looting_tool = jax.random.uniform(_rng) < 0.2
    rng, _rng = jax.random.split(rng)
    tool_id = jax.random.randint(_rng, shape=(), minval=0, maxval=2)

    is_looting_pickaxe = jnp.logical_and(
        jnp.logical_and(is_looting_tool, tool_id == 0), is_opening_chest
    )
    rng, _rng = jax.random.split(rng)
    pickaxe_loot_level = (
        jax.random.choice(
            _rng,
            (jnp.arange(4) + 1).astype(int),
            shape=(),
            p=jnp.array([0.4, 0.3, 0.2, 0.1]),
        )
        * is_looting_pickaxe
    )
    pickaxe_loot_level = jnp.maximum(pickaxe_loot_level, inventory.pickaxe)
    new_pickaxe_level = (
        is_looting_pickaxe * pickaxe_loot_level
        + (1 - is_looting_pickaxe) * inventory.pickaxe
    )

    is_looting_sword = jnp.logical_and(
        jnp.logical_and(is_looting_tool, tool_id == 1), is_opening_chest
    )
    rng, _rng = jax.random.split(rng)
    sword_loot_level = (
        jax.random.choice(
            _rng,
            (jnp.arange(4) + 1).astype(int),
            shape=(),
            p=jnp.array([0.4, 0.3, 0.2, 0.1]),
        )
        * is_looting_sword
    )
    sword_loot_level = jnp.maximum(sword_loot_level, inventory.sword)
    new_sword_level = (
        is_looting_sword * sword_loot_level + (1 - is_looting_sword) * inventory.sword
    )

    # Special chests
    is_looting_bow = jnp.logical_and(
        is_opening_chest,
        jnp.logical_and(
            state.player_level == 1,
            jnp.logical_not(state.chests_opened[state.player_level]),
        ),
    )
    new_bow_level = is_looting_bow * 1 + (1 - is_looting_bow) * inventory.bow

    is_looting_book = jnp.logical_and(
        jnp.logical_not(state.chests_opened[state.player_level]),
        jnp.logical_or(state.player_level == 3, state.player_level == 4),
    )

    # Update inventory
    return inventory.replace(
        torches=inventory.torches + torch_loot_amount * is_opening_chest,
        coal=inventory.coal + coal_loot_amount * is_opening_chest,
        iron=inventory.iron + iron_loot_amount * is_opening_chest,
        diamond=inventory.diamond + diamond_loot_amount * is_opening_chest,
        sapphire=inventory.sapphire + sapphire_loot_amount * is_opening_chest,
        ruby=inventory.ruby + ruby_loot_amount * is_opening_chest,
        arrows=inventory.arrows + arrows_loot_amount * is_opening_chest,
        pickaxe=new_pickaxe_level,
        sword=new_sword_level,
        potions=inventory.potions.at[potion_loot_index].set(
            inventory.potions[potion_loot_index]
            + potion_loot_amount * is_looting_potion * is_opening_chest
        ),
        bow=new_bow_level,
        books=inventory.books + 1 * is_looting_book * is_opening_chest,
    )


def do_action(rng, state, action, static_params):
    old_state = state

    block_position = state.player_position + DIRECTIONS[state.player_direction]

    state, did_attack_mob, did_kill_mob = attack_mob(
        state, block_position, get_player_damage_vector(state), True
    )

    # BLOCKS
    # Tree
    can_mine_tree = True
    is_block_tree = (
        state.map[state.player_level, block_position[0], block_position[1]]
        == BlockType.TREE.value
    )
    is_block_fire_tree = (
        state.map[state.player_level, block_position[0], block_position[1]]
        == BlockType.FIRE_TREE.value
    )
    is_block_ice_shrub = (
        state.map[state.player_level, block_position[0], block_position[1]]
        == BlockType.ICE_SHRUB.value
    )

    is_block_tree_type = jnp.logical_or(
        is_block_tree, jnp.logical_or(is_block_fire_tree, is_block_ice_shrub)
    )
    is_mining_tree = jnp.logical_and(
        is_block_tree_type,
        can_mine_tree,
    )
    tree_replacement_block = (
        is_block_tree * BlockType.GRASS.value
        + is_block_fire_tree * BlockType.FIRE_GRASS.value
        + is_block_ice_shrub * BlockType.ICE_GRASS.value
    )

    mined_tree_block = jax.lax.select(
        is_mining_tree,
        tree_replacement_block,
        state.map[state.player_level, block_position[0], block_position[1]],
    )
    new_map = (
        state.map[state.player_level]
        .at[block_position[0], block_position[1]]
        .set(mined_tree_block)
    )
    new_inventory = state.inventory.replace(
        wood=state.inventory.wood + 1 * is_mining_tree
    )

    # Stone
    can_mine_stone = state.inventory.pickaxe >= 1
    is_mining_stone = jnp.logical_and(
        state.map[state.player_level][block_position[0], block_position[1]]
        == BlockType.STONE.value,
        can_mine_stone,
    )
    mined_stone_block = jax.lax.select(
        is_mining_stone,
        BlockType.PATH.value,
        new_map[block_position[0], block_position[1]],
    )
    new_map = new_map.at[block_position[0], block_position[1]].set(mined_stone_block)
    new_inventory = new_inventory.replace(
        stone=new_inventory.stone + 1 * is_mining_stone
    )

    # Furnace
    is_mining_furnace = (
        state.map[state.player_level][block_position[0], block_position[1]]
        == BlockType.FURNACE.value
    )

    mined_furnace_block = jax.lax.select(
        is_mining_furnace,
        BlockType.PATH.value,
        new_map[block_position[0], block_position[1]],
    )
    new_map = new_map.at[block_position[0], block_position[1]].set(mined_furnace_block)

    # Crafting Bench
    is_mining_crafting_table = (
        state.map[state.player_level][block_position[0], block_position[1]]
        == BlockType.CRAFTING_TABLE.value
    )

    mined_crafting_table_block = jax.lax.select(
        is_mining_crafting_table,
        BlockType.PATH.value,
        new_map[block_position[0], block_position[1]],
    )
    new_map = new_map.at[block_position[0], block_position[1]].set(
        mined_crafting_table_block
    )

    # Coal
    can_mine_coal = state.inventory.pickaxe >= 1
    is_mining_coal = jnp.logical_and(
        state.map[state.player_level][block_position[0], block_position[1]]
        == BlockType.COAL.value,
        can_mine_coal,
    )
    mined_coal_block = jax.lax.select(
        is_mining_coal,
        BlockType.PATH.value,
        new_map[block_position[0], block_position[1]],
    )
    new_map = new_map.at[block_position[0], block_position[1]].set(mined_coal_block)
    new_inventory = new_inventory.replace(coal=new_inventory.coal + 1 * is_mining_coal)

    # Iron
    can_mine_iron = state.inventory.pickaxe >= 2
    is_mining_iron = jnp.logical_and(
        state.map[state.player_level][block_position[0], block_position[1]]
        == BlockType.IRON.value,
        can_mine_iron,
    )
    mined_iron_block = jax.lax.select(
        is_mining_iron,
        BlockType.PATH.value,
        new_map[block_position[0], block_position[1]],
    )
    new_map = new_map.at[block_position[0], block_position[1]].set(mined_iron_block)
    new_inventory = new_inventory.replace(iron=new_inventory.iron + 1 * is_mining_iron)

    # Diamond
    can_mine_diamond = state.inventory.pickaxe >= 3
    is_mining_diamond = jnp.logical_and(
        state.map[state.player_level][block_position[0], block_position[1]]
        == BlockType.DIAMOND.value,
        can_mine_diamond,
    )
    mined_diamond_block = jax.lax.select(
        is_mining_diamond,
        BlockType.PATH.value,
        new_map[block_position[0], block_position[1]],
    )
    new_map = new_map.at[block_position[0], block_position[1]].set(mined_diamond_block)
    new_inventory = new_inventory.replace(
        diamond=new_inventory.diamond + 1 * is_mining_diamond
    )

    # Sapphire
    can_mine_sapphire = state.inventory.pickaxe >= 4
    is_mining_sapphire = jnp.logical_and(
        state.map[state.player_level][block_position[0], block_position[1]]
        == BlockType.SAPPHIRE.value,
        can_mine_sapphire,
    )
    mined_sapphire_block = jax.lax.select(
        is_mining_sapphire,
        BlockType.PATH.value,
        new_map[block_position[0], block_position[1]],
    )
    new_map = new_map.at[block_position[0], block_position[1]].set(mined_sapphire_block)
    new_inventory = new_inventory.replace(
        sapphire=new_inventory.sapphire + 1 * is_mining_sapphire
    )

    # Ruby
    can_mine_ruby = state.inventory.pickaxe >= 4
    is_mining_ruby = jnp.logical_and(
        state.map[state.player_level][block_position[0], block_position[1]]
        == BlockType.RUBY.value,
        can_mine_ruby,
    )
    mined_ruby_block = jax.lax.select(
        is_mining_ruby,
        BlockType.PATH.value,
        new_map[block_position[0], block_position[1]],
    )
    new_map = new_map.at[block_position[0], block_position[1]].set(mined_ruby_block)
    new_inventory = new_inventory.replace(ruby=new_inventory.ruby + 1 * is_mining_ruby)

    # Sapling
    rng, _rng = jax.random.split(rng)
    is_mining_sapling = jnp.logical_and(
        state.map[state.player_level][block_position[0], block_position[1]]
        == BlockType.GRASS.value,
        jax.random.uniform(_rng) < 0.1,
    )

    new_inventory = new_inventory.replace(
        sapling=new_inventory.sapling + 1 * is_mining_sapling
    )

    # Water
    is_drinking_water = jnp.logical_or(
        state.map[state.player_level][block_position[0], block_position[1]]
        == BlockType.WATER.value,
        state.map[state.player_level][block_position[0], block_position[1]]
        == BlockType.FOUNTAIN.value,
    )
    new_drink = jax.lax.select(
        is_drinking_water,
        jnp.minimum(get_max_drink(state), state.player_drink + 1),
        state.player_drink,
    )
    new_thirst = jax.lax.select(is_drinking_water, 0.0, state.player_thirst)
    new_achievements = state.achievements.at[Achievement.COLLECT_DRINK.value].set(
        jnp.logical_or(
            state.achievements[Achievement.COLLECT_DRINK.value], is_drinking_water
        )
    )

    # Plant
    is_eating_plant = (
        state.map[state.player_level][block_position[0], block_position[1]]
        == BlockType.RIPE_PLANT.value
    )
    new_plant = jax.lax.select(
        is_eating_plant,
        BlockType.PLANT.value,
        new_map[block_position[0], block_position[1]],
    )
    new_map = new_map.at[block_position[0], block_position[1]].set(new_plant)
    new_food = jax.lax.select(
        is_eating_plant,
        jnp.minimum(get_max_food(state), state.player_food + 4),
        state.player_food,
    )
    new_hunger = jax.lax.select(is_eating_plant, 0.0, state.player_hunger)
    new_achievements = new_achievements.at[Achievement.EAT_PLANT.value].set(
        jnp.logical_or(new_achievements[Achievement.EAT_PLANT.value], is_eating_plant)
    )
    new_growing_plants_age = update_plants_with_eat(
        state, block_position, static_params
    )

    # Stalagmite
    can_mine_stalagmite = state.inventory.pickaxe >= 1
    is_mining_stalagmite = jnp.logical_and(
        state.map[state.player_level][block_position[0], block_position[1]]
        == BlockType.STALAGMITE.value,
        can_mine_stalagmite,
    )
    mined_stalagmite_block = jax.lax.select(
        is_mining_stalagmite,
        BlockType.PATH.value,
        new_map[block_position[0], block_position[1]],
    )
    new_map = new_map.at[block_position[0], block_position[1]].set(
        mined_stalagmite_block
    )
    new_inventory = new_inventory.replace(
        stone=new_inventory.stone + 1 * is_mining_stalagmite
    )

    # Chest
    is_opening_chest = (
        state.map[state.player_level][block_position[0], block_position[1]]
        == BlockType.CHEST.value
    )
    mined_chest_block = jax.lax.select(
        is_opening_chest,
        BlockType.PATH.value,
        new_map[block_position[0], block_position[1]],
    )
    new_map = new_map.at[block_position[0], block_position[1]].set(mined_chest_block)
    rng, _rng = jax.random.split(rng)
    new_inventory = add_items_from_chest(_rng, state, new_inventory, is_opening_chest)

    new_chests_opened = state.chests_opened.at[state.player_level].set(
        jnp.logical_or(state.chests_opened[state.player_level], is_opening_chest)
    )

    new_achievements = new_achievements.at[Achievement.OPEN_CHEST.value].set(
        jnp.logical_or(
            state.achievements[Achievement.OPEN_CHEST.value], is_opening_chest
        )
    )

    # Boss
    is_attacking_boss = (
        state.map[state.player_level][block_position[0], block_position[1]]
        == BlockType.NECROMANCER.value
    )

    is_damaging_boss = jnp.logical_and(
        is_attacking_boss,
        jnp.logical_and(
            is_boss_vulnerable(state), is_fighting_boss(state, static_params)
        ),
    )

    new_boss_progress = state.boss_progress + 1 * is_damaging_boss
    new_boss_timesteps_to_spawn_this_round = (
        BOSS_FIGHT_SPAWN_TURNS * is_damaging_boss
        + state.boss_timesteps_to_spawn_this_round * (1 - is_damaging_boss)
    )

    new_achievements = new_achievements.at[Achievement.DAMAGE_NECROMANCER.value].set(
        jnp.logical_or(
            new_achievements[Achievement.DAMAGE_NECROMANCER.value], is_damaging_boss
        )
    )

    # Action mining
    action_block_in_bounds = in_bounds(state, block_position)
    action_block_in_bounds = jnp.logical_and(
        action_block_in_bounds, jnp.logical_not(did_attack_mob)
    )
    new_map = jax.lax.select(
        action_block_in_bounds, new_map, state.map[state.player_level]
    )
    new_inventory = jax.tree.map(
        lambda x, y: jax.lax.select(action_block_in_bounds, x, y),
        new_inventory,
        state.inventory,
    )
    new_drink = jax.lax.select(action_block_in_bounds, new_drink, state.player_drink)
    new_thirst = jax.lax.select(action_block_in_bounds, new_thirst, state.player_thirst)
    new_food = jax.lax.select(action_block_in_bounds, new_food, state.player_food)
    new_hunger = jax.lax.select(action_block_in_bounds, new_hunger, state.player_hunger)
    new_growing_plants_age = jax.lax.select(
        action_block_in_bounds, new_growing_plants_age, state.growing_plants_age
    )

    new_achievements = jax.lax.select(
        action_block_in_bounds, new_achievements, state.achievements
    )

    new_whole_map = state.map.at[state.player_level].set(new_map)

    state = state.replace(
        map=new_whole_map,
        inventory=new_inventory,
        player_drink=new_drink,
        player_thirst=new_thirst,
        player_food=new_food,
        player_hunger=new_hunger,
        growing_plants_age=new_growing_plants_age,
        achievements=new_achievements,
        chests_opened=new_chests_opened,
        boss_progress=new_boss_progress,
        boss_timesteps_to_spawn_this_round=new_boss_timesteps_to_spawn_this_round,
    )

    # Do?
    doing_mining = action == Action.DO.value
    state = jax.tree.map(
        lambda x, y: jax.lax.select(doing_mining, x, y),
        state,
        old_state,
    )

    return state


def do_crafting(state, action):
    is_at_crafting_table = is_near_block(state, BlockType.CRAFTING_TABLE.value)
    is_at_furnace = is_near_block(state, BlockType.FURNACE.value)

    new_achievements = state.achievements

    # Wood pickaxe
    can_craft_wood_pickaxe = state.inventory.wood >= 1

    is_crafting_wood_pickaxe = jnp.logical_and(
        action == Action.MAKE_WOOD_PICKAXE.value,
        jnp.logical_and(
            can_craft_wood_pickaxe,
            jnp.logical_and(is_at_crafting_table, state.inventory.pickaxe < 1),
        ),
    )

    new_inventory = state.inventory.replace(
        wood=state.inventory.wood - 1 * is_crafting_wood_pickaxe,
        pickaxe=state.inventory.pickaxe * (1 - is_crafting_wood_pickaxe)
        + 1 * is_crafting_wood_pickaxe,
    )

    # Stone pickaxe
    can_craft_stone_pickaxe = jnp.logical_and(
        new_inventory.wood >= 1, new_inventory.stone >= 1
    )
    is_crafting_stone_pickaxe = jnp.logical_and(
        action == Action.MAKE_STONE_PICKAXE.value,
        jnp.logical_and(
            can_craft_stone_pickaxe,
            jnp.logical_and(is_at_crafting_table, new_inventory.pickaxe < 2),
        ),
    )

    new_inventory = new_inventory.replace(
        stone=new_inventory.stone - 1 * is_crafting_stone_pickaxe,
        wood=new_inventory.wood - 1 * is_crafting_stone_pickaxe,
        pickaxe=new_inventory.pickaxe * (1 - is_crafting_stone_pickaxe)
        + 2 * is_crafting_stone_pickaxe,
    )

    # Iron pickaxe
    can_craft_iron_pickaxe = jnp.logical_and(
        new_inventory.wood >= 1,
        jnp.logical_and(
            new_inventory.stone >= 1,
            jnp.logical_and(
                new_inventory.iron >= 1,
                new_inventory.coal >= 1,
            ),
        ),
    )
    is_crafting_iron_pickaxe = jnp.logical_and(
        action == Action.MAKE_IRON_PICKAXE.value,
        jnp.logical_and(
            can_craft_iron_pickaxe,
            jnp.logical_and(
                is_at_furnace,
                jnp.logical_and(is_at_crafting_table, new_inventory.pickaxe < 3),
            ),
        ),
    )

    new_inventory = new_inventory.replace(
        iron=new_inventory.iron - 1 * is_crafting_iron_pickaxe,
        wood=new_inventory.wood - 1 * is_crafting_iron_pickaxe,
        stone=new_inventory.stone - 1 * is_crafting_iron_pickaxe,
        coal=new_inventory.coal - 1 * is_crafting_iron_pickaxe,
        pickaxe=new_inventory.pickaxe * (1 - is_crafting_iron_pickaxe)
        + 3 * is_crafting_iron_pickaxe,
    )

    # Diamond pickaxe
    can_craft_diamond_pickaxe = jnp.logical_and(
        new_inventory.wood >= 1, new_inventory.diamond >= 3
    )
    is_crafting_diamond_pickaxe = jnp.logical_and(
        action == Action.MAKE_DIAMOND_PICKAXE.value,
        jnp.logical_and(
            can_craft_diamond_pickaxe,
            jnp.logical_and(is_at_crafting_table, new_inventory.pickaxe < 4),
        ),
    )

    new_inventory = new_inventory.replace(
        diamond=new_inventory.diamond - 3 * is_crafting_diamond_pickaxe,
        wood=new_inventory.wood - 1 * is_crafting_diamond_pickaxe,
        pickaxe=new_inventory.pickaxe * (1 - is_crafting_diamond_pickaxe)
        + 4 * is_crafting_diamond_pickaxe,
    )

    # Wood sword
    can_craft_wood_sword = new_inventory.wood >= 1
    is_crafting_wood_sword = jnp.logical_and(
        action == Action.MAKE_WOOD_SWORD.value,
        jnp.logical_and(
            can_craft_wood_sword,
            jnp.logical_and(is_at_crafting_table, new_inventory.sword < 1),
        ),
    )

    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_wood_sword,
        sword=new_inventory.sword * (1 - is_crafting_wood_sword)
        + 1 * is_crafting_wood_sword,
    )

    # Stone sword
    can_craft_stone_sword = jnp.logical_and(
        new_inventory.stone >= 1, new_inventory.wood >= 1
    )
    is_crafting_stone_sword = jnp.logical_and(
        action == Action.MAKE_STONE_SWORD.value,
        jnp.logical_and(
            can_craft_stone_sword,
            jnp.logical_and(is_at_crafting_table, new_inventory.sword < 2),
        ),
    )

    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_stone_sword,
        stone=new_inventory.stone - 1 * is_crafting_stone_sword,
        sword=new_inventory.sword * (1 - is_crafting_stone_sword)
        + 2 * is_crafting_stone_sword,
    )

    # Iron sword
    can_craft_iron_sword = jnp.logical_and(
        new_inventory.iron >= 1,
        jnp.logical_and(
            new_inventory.wood >= 1,
            jnp.logical_and(new_inventory.stone >= 1, new_inventory.coal >= 1),
        ),
    )
    is_crafting_iron_sword = jnp.logical_and(
        action == Action.MAKE_IRON_SWORD.value,
        jnp.logical_and(
            can_craft_iron_sword,
            jnp.logical_and(
                is_at_furnace,
                jnp.logical_and(is_at_crafting_table, new_inventory.sword < 3),
            ),
        ),
    )

    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_iron_sword,
        iron=new_inventory.iron - 1 * is_crafting_iron_sword,
        stone=new_inventory.stone - 1 * is_crafting_iron_sword,
        coal=new_inventory.coal - 1 * is_crafting_iron_sword,
        sword=new_inventory.sword * (1 - is_crafting_iron_sword)
        + 3 * is_crafting_iron_sword,
    )

    # Diamond sword
    can_craft_diamond_sword = jnp.logical_and(
        new_inventory.diamond >= 2, new_inventory.wood >= 1
    )
    is_crafting_diamond_sword = jnp.logical_and(
        action == Action.MAKE_DIAMOND_SWORD.value,
        jnp.logical_and(
            can_craft_diamond_sword,
            jnp.logical_and(is_at_crafting_table, new_inventory.sword < 4),
        ),
    )

    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_diamond_sword,
        diamond=new_inventory.diamond - 2 * is_crafting_diamond_sword,
        sword=new_inventory.sword * (1 - is_crafting_diamond_sword)
        + 4 * is_crafting_diamond_sword,
    )

    # Iron armour
    can_craft_iron_armour = (new_inventory.armour < 1).sum() > 0
    can_craft_iron_armour = jnp.logical_and(
        can_craft_iron_armour,
        jnp.logical_and(new_inventory.iron >= 3, new_inventory.coal >= 3),
    )

    iron_armour_index_to_craft = jnp.argmax(new_inventory.armour < 1)

    is_crafting_iron_armour = jnp.logical_and(
        action == Action.MAKE_IRON_ARMOUR.value,
        jnp.logical_and(
            can_craft_iron_armour,
            jnp.logical_and(is_at_crafting_table, is_at_furnace),
        ),
    )

    new_inventory = new_inventory.replace(
        iron=new_inventory.iron - 3 * is_crafting_iron_armour,
        coal=new_inventory.coal - 3 * is_crafting_iron_armour,
        armour=new_inventory.armour.at[iron_armour_index_to_craft].set(
            is_crafting_iron_armour * 1
            + (1 - is_crafting_iron_armour)
            * new_inventory.armour[iron_armour_index_to_craft]
        ),
    )
    new_achievements = new_achievements.at[Achievement.MAKE_IRON_ARMOUR.value].set(
        jnp.logical_or(
            new_achievements[Achievement.MAKE_IRON_ARMOUR.value],
            is_crafting_iron_armour,
        )
    )

    # Diamond armour
    can_craft_diamond_armour = (new_inventory.armour < 2).sum() > 0
    can_craft_diamond_armour = jnp.logical_and(
        can_craft_diamond_armour, new_inventory.diamond >= 3
    )

    diamond_armour_index_to_craft = jnp.argmax(new_inventory.armour < 2)

    is_crafting_diamond_armour = jnp.logical_and(
        action == Action.MAKE_DIAMOND_ARMOUR.value,
        jnp.logical_and(
            can_craft_diamond_armour,
            is_at_crafting_table,
        ),
    )

    new_inventory = new_inventory.replace(
        diamond=new_inventory.diamond - 3 * is_crafting_diamond_armour,
        armour=new_inventory.armour.at[diamond_armour_index_to_craft].set(
            is_crafting_diamond_armour * 2
            + (1 - is_crafting_diamond_armour)
            * new_inventory.armour[diamond_armour_index_to_craft]
        ),
    )
    new_achievements = new_achievements.at[Achievement.MAKE_DIAMOND_ARMOUR.value].set(
        jnp.logical_or(
            new_achievements[Achievement.MAKE_DIAMOND_ARMOUR.value],
            is_crafting_diamond_armour,
        )
    )

    # Arrow
    can_craft_arrow = jnp.logical_and(new_inventory.stone >= 1, new_inventory.wood >= 1)
    is_crafting_arrow = jnp.logical_and(
        action == Action.MAKE_ARROW.value,
        jnp.logical_and(
            can_craft_arrow,
            jnp.logical_and(is_at_crafting_table, new_inventory.arrows < 99),
        ),
    )
    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_arrow,
        stone=new_inventory.stone - 1 * is_crafting_arrow,
        arrows=new_inventory.arrows + 2 * is_crafting_arrow,
    )

    # Torch
    can_craft_torch = jnp.logical_and(new_inventory.coal >= 1, new_inventory.wood >= 1)
    is_crafting_torch = jnp.logical_and(
        action == Action.MAKE_TORCH.value,
        jnp.logical_and(
            can_craft_torch,
            jnp.logical_and(is_at_crafting_table, new_inventory.torches < 99),
        ),
    )
    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_torch,
        coal=new_inventory.coal - 1 * is_crafting_torch,
        torches=new_inventory.torches + 4 * is_crafting_torch,
    )

    state = state.replace(
        inventory=new_inventory,
        achievements=new_achievements,
    )

    return state


def add_new_growing_plant(state, position, is_placing_sapling, static_params):
    def _is_empty(unused, index):
        return None, jnp.logical_not(state.growing_plants_mask[index])

    _, is_empty = jax.lax.scan(
        _is_empty, None, jnp.arange(static_params.max_growing_plants)
    )

    plant_index = jnp.argmax(is_empty)
    is_an_empty_slot = is_empty.sum() > 0

    is_adding_plant = jnp.logical_and(is_an_empty_slot, is_placing_sapling)

    new_growing_plants_positions = jax.lax.select(
        is_adding_plant,
        state.growing_plants_positions.at[plant_index].set(position),
        state.growing_plants_positions,
    )
    new_growing_plants_age = jax.lax.select(
        is_adding_plant,
        state.growing_plants_age.at[plant_index].set(0),
        state.growing_plants_age,
    )
    new_growing_plants_mask = jax.lax.select(
        is_adding_plant,
        state.growing_plants_mask.at[plant_index].set(True),
        state.growing_plants_mask,
    )

    return new_growing_plants_positions, new_growing_plants_age, new_growing_plants_mask


def place_block(state, action, static_params):
    placing_block_position = state.player_position + DIRECTIONS[state.player_direction]

    new_item_map = state.item_map[state.player_level]

    is_placement_on_solid_block_or_item = jnp.logical_or(
        is_in_solid_block(state, placing_block_position),
        new_item_map[placing_block_position[0], placing_block_position[1]]
        != ItemType.NONE.value,
    )

    # Crafting table
    crafting_table_key_down = action == Action.PLACE_TABLE.value
    has_wood = state.inventory.wood >= 2
    is_placing_crafting_table = jnp.logical_and(
        crafting_table_key_down,
        jnp.logical_and(jnp.logical_not(is_placement_on_solid_block_or_item), has_wood),
    )
    placed_crafting_table_block = jax.lax.select(
        is_placing_crafting_table,
        BlockType.CRAFTING_TABLE.value,
        state.map[state.player_level][
            placing_block_position[0], placing_block_position[1]
        ],
    )
    new_map = (
        state.map[state.player_level]
        .at[placing_block_position[0], placing_block_position[1]]
        .set(placed_crafting_table_block)
    )
    new_inventory = state.inventory.replace(
        wood=state.inventory.wood - 2 * is_placing_crafting_table
    )
    new_achievements = state.achievements.at[Achievement.PLACE_TABLE.value].set(
        jnp.logical_or(
            state.achievements[Achievement.PLACE_TABLE.value], is_placing_crafting_table
        )
    )

    # Furnace
    furnace_key_down = action == Action.PLACE_FURNACE.value
    has_stone = new_inventory.stone > 0
    is_placing_furnace = jnp.logical_and(
        furnace_key_down,
        jnp.logical_and(
            jnp.logical_not(is_placement_on_solid_block_or_item), has_stone
        ),
    )
    placed_furnace_block = jax.lax.select(
        is_placing_furnace,
        BlockType.FURNACE.value,
        new_map[placing_block_position[0], placing_block_position[1]],
    )
    new_map = new_map.at[placing_block_position[0], placing_block_position[1]].set(
        placed_furnace_block
    )
    new_inventory = new_inventory.replace(
        stone=new_inventory.stone - 1 * is_placing_furnace
    )
    new_achievements = new_achievements.at[Achievement.PLACE_FURNACE.value].set(
        jnp.logical_or(
            new_achievements[Achievement.PLACE_FURNACE.value], is_placing_furnace
        )
    )

    # Stone
    stone_key_down = action == Action.PLACE_STONE.value
    has_stone = new_inventory.stone > 0
    is_placing_on_valid_block = jnp.logical_or(
        state.map[state.player_level][
            placing_block_position[0], placing_block_position[1]
        ]
        == BlockType.WATER.value,
        jnp.logical_not(is_placement_on_solid_block_or_item),
    )
    is_placing_stone = jnp.logical_and(
        stone_key_down,
        jnp.logical_and(is_placing_on_valid_block, has_stone),
    )
    placed_stone_block = jax.lax.select(
        is_placing_stone,
        BlockType.STONE.value,
        new_map[placing_block_position[0], placing_block_position[1]],
    )
    new_map = new_map.at[placing_block_position[0], placing_block_position[1]].set(
        placed_stone_block
    )
    new_inventory = new_inventory.replace(
        stone=new_inventory.stone - 1 * is_placing_stone
    )
    new_achievements = new_achievements.at[Achievement.PLACE_STONE.value].set(
        jnp.logical_or(
            new_achievements[Achievement.PLACE_STONE.value], is_placing_stone
        )
    )

    # Torch
    torch_key_down = action == Action.PLACE_TORCH.value
    has_torch = new_inventory.torches > 0

    is_placing_on_valid_block = CAN_PLACE_ITEM_MAPPING[
        state.map[state.player_level][
            placing_block_position[0], placing_block_position[1]
        ]
    ]

    is_placing_on_valid_block = jnp.logical_and(
        is_placing_on_valid_block,
        new_item_map[placing_block_position[0], placing_block_position[1]]
        == ItemType.NONE.value,
    )
    is_placing_torch = jnp.logical_and(
        torch_key_down,
        jnp.logical_and(is_placing_on_valid_block, has_torch),
    )

    placed_torch_item = jax.lax.select(
        is_placing_torch,
        ItemType.TORCH.value,
        new_item_map[placing_block_position[0], placing_block_position[1]],
    )
    new_item_map = new_item_map.at[
        placing_block_position[0], placing_block_position[1]
    ].set(placed_torch_item)
    new_inventory = new_inventory.replace(
        torches=new_inventory.torches - 1 * is_placing_torch
    )

    light_map_padding = 6
    padded_light_map = jnp.pad(
        state.light_map[state.player_level],
        (light_map_padding, light_map_padding),
        constant_values=0,
    )

    current_light_map = jax.lax.dynamic_slice(
        padded_light_map,
        placing_block_position
        - jnp.array([4, 4])
        + jnp.array([light_map_padding, light_map_padding]),
        (9, 9),
    )

    torch_light_map = jnp.clip(TORCH_LIGHT_MAP + current_light_map, 0.0, 1.0)

    torch_light_map = torch_light_map * is_placing_torch + current_light_map * (
        1 - is_placing_torch
    )

    new_padded_light_map_floor = jax.lax.dynamic_update_slice(
        padded_light_map,
        torch_light_map,
        placing_block_position
        - jnp.array([4, 4])
        + jnp.array([light_map_padding, light_map_padding]),
    )
    new_light_map_floor = new_padded_light_map_floor[
        light_map_padding:-light_map_padding, light_map_padding:-light_map_padding
    ]
    new_light_map = state.light_map.at[state.player_level].set(new_light_map_floor)

    new_achievements = new_achievements.at[Achievement.PLACE_TORCH.value].set(
        jnp.logical_or(
            new_achievements[Achievement.PLACE_TORCH.value], is_placing_torch
        )
    )

    # Plant
    sapling_key_down = action == Action.PLACE_PLANT.value
    has_sapling = new_inventory.sapling > 0
    is_placing_sapling = jnp.logical_and(
        sapling_key_down,
        jnp.logical_and(
            new_map[placing_block_position[0], placing_block_position[1]]
            == BlockType.GRASS.value,
            has_sapling,
        ),
    )
    is_placing_sapling = jnp.logical_and(
        is_placing_sapling,
        new_item_map[placing_block_position[0], placing_block_position[1]]
        == ItemType.NONE.value,
    )
    placed_sapling_block = jax.lax.select(
        is_placing_sapling,
        BlockType.PLANT.value,
        new_map[placing_block_position[0], placing_block_position[1]],
    )
    new_map = new_map.at[placing_block_position[0], placing_block_position[1]].set(
        placed_sapling_block
    )
    new_inventory = new_inventory.replace(
        sapling=new_inventory.sapling - 1 * is_placing_sapling
    )
    new_achievements = new_achievements.at[Achievement.PLACE_PLANT.value].set(
        jnp.logical_or(
            new_achievements[Achievement.PLACE_PLANT.value], is_placing_sapling
        )
    )
    (
        new_growing_plants_positions,
        new_growing_plants_age,
        new_growing_plants_mask,
    ) = add_new_growing_plant(
        state, placing_block_position, is_placing_sapling, static_params
    )

    # Do?

    action_block = state.player_position + DIRECTIONS[state.player_direction]
    action_block_in_bounds = in_bounds(state, action_block)
    action_block_in_bounds = jnp.logical_and(
        action_block_in_bounds, jnp.logical_not(is_in_mob(state, action_block))
    )

    new_map = jax.lax.select(
        action_block_in_bounds, new_map, state.map[state.player_level]
    )
    new_item_map = jax.lax.select(
        action_block_in_bounds, new_item_map, state.item_map[state.player_level]
    )
    new_inventory = jax.tree.map(
        lambda x, y: jax.lax.select(action_block_in_bounds, x, y),
        new_inventory,
        state.inventory,
    )
    new_achievements = jax.tree.map(
        lambda x, y: jax.lax.select(action_block_in_bounds, x, y),
        new_achievements,
        state.achievements,
    )
    new_growing_plants_positions = jax.lax.select(
        action_block_in_bounds,
        new_growing_plants_positions,
        state.growing_plants_positions,
    )
    new_growing_plants_age = jax.lax.select(
        action_block_in_bounds, new_growing_plants_age, state.growing_plants_age
    )
    new_growing_plants_mask = jax.lax.select(
        action_block_in_bounds, new_growing_plants_mask, state.growing_plants_mask
    )
    new_light_map = jax.lax.select(
        action_block_in_bounds, new_light_map, state.light_map
    )

    new_whole_map = state.map.at[state.player_level].set(new_map)
    new_whole_item_map = state.item_map.at[state.player_level].set(new_item_map)

    state = state.replace(
        map=new_whole_map,
        item_map=new_whole_item_map,
        light_map=new_light_map,
        inventory=new_inventory,
        achievements=new_achievements,
        growing_plants_positions=new_growing_plants_positions,
        growing_plants_age=new_growing_plants_age,
        growing_plants_mask=new_growing_plants_mask,
    )

    return state


def update_mobs(rng, state, params, static_params):

    # Move melee_mobs
    def _move_melee_mob(rng_and_state, melee_mob_index):
        rng, state = rng_and_state
        melee_mobs = state.melee_mobs

        # Random move
        rng, _rng = jax.random.split(rng)
        random_move_direction = jax.random.choice(
            _rng,
            DIRECTIONS[1:5],
        )
        random_move_proposed_position = (
            melee_mobs.position[state.player_level, melee_mob_index]
            + random_move_direction
        )

        # Move towards player
        player_move_direction = jnp.zeros((2,), dtype=jnp.int32)
        player_move_direction_abs = jnp.abs(
            state.player_position
            - melee_mobs.position[state.player_level, melee_mob_index]
        )
        player_move_direction_index_p = (
            player_move_direction_abs == player_move_direction_abs.max()
        ) / player_move_direction_abs.sum()
        rng, _rng = jax.random.split(rng)
        player_move_direction_index = jax.random.choice(
            _rng,
            jnp.arange(2),
            p=player_move_direction_index_p,
        )

        player_move_direction = player_move_direction.at[
            player_move_direction_index
        ].set(
            jnp.sign(
                state.player_position[player_move_direction_index]
                - melee_mobs.position[state.player_level, melee_mob_index][
                    player_move_direction_index
                ]
            ).astype(jnp.int32)
        )
        player_move_proposed_position = (
            melee_mobs.position[state.player_level, melee_mob_index]
            + player_move_direction
        )

        # Choose movement
        close_to_player = (
            jnp.sum(
                jnp.abs(
                    melee_mobs.position[state.player_level, melee_mob_index]
                    - state.player_position
                )
            )
            < 10
        )
        close_to_player = jnp.logical_or(
            close_to_player, is_fighting_boss(state, static_params)
        )

        rng, _rng = jax.random.split(rng)
        close_to_player = jnp.logical_and(
            close_to_player, jax.random.uniform(_rng) < 0.75
        )

        proposed_position = jax.lax.select(
            close_to_player,
            player_move_proposed_position,
            random_move_proposed_position,
        )

        # Choose attack or not
        is_attacking_player = (
            jnp.sum(
                jnp.abs(
                    melee_mobs.position[state.player_level, melee_mob_index]
                    - state.player_position
                )
            )
            == 1
        )
        is_attacking_player = jnp.logical_and(
            is_attacking_player,
            melee_mobs.attack_cooldown[state.player_level, melee_mob_index] <= 0,
        )
        is_attacking_player = jnp.logical_and(
            is_attacking_player, melee_mobs.mask[state.player_level, melee_mob_index]
        )

        proposed_position = jax.lax.select(
            is_attacking_player,
            melee_mobs.position[state.player_level, melee_mob_index],
            proposed_position,
        )

        melee_mob_base_damage = MOB_TYPE_DAMAGE_MAPPING[
            melee_mobs.type_id[state.player_level, melee_mob_index], MobType.MELEE.value
        ]

        melee_mob_damage = get_damage_done_to_player(
            state, static_params, melee_mob_base_damage * (1 + 2.5 * state.is_sleeping)
        )

        new_cooldown = jax.lax.select(
            is_attacking_player,
            5,
            melee_mobs.attack_cooldown[state.player_level, melee_mob_index] - 1,
        )

        is_waking_player = jnp.logical_and(state.is_sleeping, is_attacking_player)

        state = state.replace(
            player_health=state.player_health - melee_mob_damage * is_attacking_player,
            is_sleeping=jnp.logical_and(
                state.is_sleeping, jnp.logical_not(is_attacking_player)
            ),
            is_resting=jnp.logical_and(
                state.is_resting, jnp.logical_not(is_attacking_player)
            ),
            achievements=state.achievements.at[Achievement.WAKE_UP.value].set(
                jnp.logical_or(
                    state.achievements[Achievement.WAKE_UP.value], is_waking_player
                )
            ),
        )

        mob_type = melee_mobs.type_id[state.player_level, melee_mob_index]
        collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type, 1]
        valid_move = is_position_in_bounds_not_in_mob_not_colliding(
            state, proposed_position, collision_map
        )
        position = jax.lax.select(
            valid_move,
            proposed_position,
            melee_mobs.position[state.player_level, melee_mob_index],
        )

        should_not_despawn = (
            jnp.abs(
                melee_mobs.position[state.player_level, melee_mob_index]
                - state.player_position
            ).sum()
            < params.mob_despawn_distance
        )
        should_not_despawn = jnp.logical_or(
            should_not_despawn, is_fighting_boss(state, static_params)
        )

        rng, _rng = jax.random.split(rng)

        # Clear our old entry if we are alive
        new_mob_map = state.mob_map.at[
            state.player_level,
            state.melee_mobs.position[state.player_level, melee_mob_index, 0],
            state.melee_mobs.position[state.player_level, melee_mob_index, 1],
        ].set(
            jnp.logical_and(
                state.mob_map[
                    state.player_level,
                    state.melee_mobs.position[state.player_level, melee_mob_index, 0],
                    state.melee_mobs.position[state.player_level, melee_mob_index, 1],
                ],
                jnp.logical_not(melee_mobs.mask[state.player_level, melee_mob_index]),
            )
        )
        new_mask = jnp.logical_and(
            state.melee_mobs.mask[state.player_level, melee_mob_index],
            should_not_despawn,
        )
        # Enter new entry if we are alive and not despawning this timestep
        new_mob_map = new_mob_map.at[state.player_level, position[0], position[1]].set(
            jnp.logical_or(
                new_mob_map[state.player_level, position[0], position[1]], new_mask
            )
        )

        state = state.replace(
            melee_mobs=state.melee_mobs.replace(
                position=state.melee_mobs.position.at[
                    state.player_level, melee_mob_index
                ].set(position),
                attack_cooldown=state.melee_mobs.attack_cooldown.at[
                    state.player_level, melee_mob_index
                ].set(new_cooldown),
                mask=state.melee_mobs.mask.at[state.player_level, melee_mob_index].set(
                    new_mask
                ),
            ),
            mob_map=new_mob_map,
        )

        return (_rng, state), None

    rng, _rng = jax.random.split(rng)
    (rng, state), _ = jax.lax.scan(
        _move_melee_mob, (rng, state), jnp.arange(static_params.max_melee_mobs)
    )

    # Move passive_mobs
    def _move_passive_mob(rng_and_state, passive_mob_index):
        rng, state = rng_and_state
        passive_mobs = state.passive_mobs

        # Random move
        rng, _rng = jax.random.split(rng)
        random_move_direction = jax.random.choice(
            _rng,
            DIRECTIONS[1:9],  # 50% chance of not moving
        )
        proposed_position = (
            passive_mobs.position[state.player_level, passive_mob_index]
            + random_move_direction
        )

        mob_type = passive_mobs.type_id[state.player_level, passive_mob_index]
        collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type, 0]
        valid_move = is_position_in_bounds_not_in_mob_not_colliding(
            state, proposed_position, collision_map
        )
        position = jax.lax.select(
            valid_move,
            proposed_position,
            passive_mobs.position[state.player_level, passive_mob_index],
        )

        should_not_despawn = (
            jnp.abs(
                passive_mobs.position[state.player_level, passive_mob_index]
                - state.player_position
            ).sum()
            < params.mob_despawn_distance
        )

        # Clear our old entry if we are alive
        new_mob_map = state.mob_map.at[
            state.player_level,
            state.passive_mobs.position[state.player_level, passive_mob_index, 0],
            state.passive_mobs.position[state.player_level, passive_mob_index, 1],
        ].set(
            jnp.logical_and(
                state.mob_map[
                    state.player_level,
                    state.passive_mobs.position[
                        state.player_level, passive_mob_index, 0
                    ],
                    state.passive_mobs.position[
                        state.player_level, passive_mob_index, 1
                    ],
                ],
                jnp.logical_not(
                    passive_mobs.mask[state.player_level, passive_mob_index]
                ),
            )
        )
        new_mask = jnp.logical_and(
            state.passive_mobs.mask[state.player_level, passive_mob_index],
            should_not_despawn,
        )
        # Enter new entry if we are alive and not despawning this timestep
        new_mob_map = new_mob_map.at[state.player_level, position[0], position[1]].set(
            jnp.logical_or(
                new_mob_map[state.player_level, position[0], position[1]], new_mask
            )
        )

        state = state.replace(
            passive_mobs=state.passive_mobs.replace(
                position=state.passive_mobs.position.at[
                    state.player_level, passive_mob_index
                ].set(position),
                mask=state.passive_mobs.mask.at[
                    state.player_level, passive_mob_index
                ].set(
                    jnp.logical_and(
                        state.passive_mobs.mask[state.player_level, passive_mob_index],
                        should_not_despawn,
                    )
                ),
            ),
            mob_map=new_mob_map,
        )

        return (rng, state), None

    rng, _rng = jax.random.split(rng)
    (rng, state), _ = jax.lax.scan(
        _move_passive_mob, (rng, state), jnp.arange(static_params.max_passive_mobs)
    )

    # Move ranged_mobs

    def _move_ranged_mob(rng_and_state, ranged_mob_index):
        rng, state = rng_and_state
        ranged_mobs = state.ranged_mobs

        # Random move
        rng, _rng = jax.random.split(rng)
        random_move_direction = jax.random.choice(
            _rng,
            DIRECTIONS[1:5],
        )
        random_move_proposed_position = (
            ranged_mobs.position[state.player_level, ranged_mob_index]
            + random_move_direction
        )

        # Move towards player
        player_move_direction = jnp.zeros((2,), dtype=jnp.int32)
        player_move_direction_abs = jnp.abs(
            state.player_position
            - ranged_mobs.position[state.player_level, ranged_mob_index]
        )
        player_move_direction_index_p = (
            player_move_direction_abs == player_move_direction_abs.max()
        ) / player_move_direction_abs.sum()
        rng, _rng = jax.random.split(rng)
        player_move_direction_index = jax.random.choice(
            _rng,
            jnp.arange(2),
            p=player_move_direction_index_p,
        )

        player_move_direction = player_move_direction.at[
            player_move_direction_index
        ].set(
            jnp.sign(
                state.player_position[player_move_direction_index]
                - ranged_mobs.position[state.player_level, ranged_mob_index][
                    player_move_direction_index
                ]
            ).astype(jnp.int32)
        )
        player_move_towards_proposed_position = (
            ranged_mobs.position[state.player_level, ranged_mob_index]
            + player_move_direction
        )
        player_move_away_proposed_position = (
            ranged_mobs.position[state.player_level, ranged_mob_index]
            - player_move_direction
        )

        # Choose movement
        distance_to_player = jnp.sum(
            jnp.abs(
                ranged_mobs.position[state.player_level, ranged_mob_index]
                - state.player_position
            )
        )

        far_from_player = distance_to_player >= 6
        too_close_to_player = distance_to_player <= 3

        proposed_position = jax.lax.select(
            far_from_player,
            player_move_towards_proposed_position,
            random_move_proposed_position,
        )
        proposed_position = jax.lax.select(
            too_close_to_player,
            player_move_away_proposed_position,
            proposed_position,
        )

        rng, _rng = jax.random.split(rng)

        proposed_position = jax.lax.select(
            jax.random.uniform(_rng) > 0.85,
            proposed_position,
            random_move_proposed_position,
        )

        # Choose attack or not
        is_attacking_player = jnp.logical_and(
            distance_to_player >= 4, distance_to_player <= 5
        )
        # If we're too close to player (so we want to run) but are blocked, we shoot
        mob_type = ranged_mobs.type_id[state.player_level, ranged_mob_index]
        collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type, 2]
        is_attacking_player = jnp.logical_or(
            is_attacking_player,
            jnp.logical_and(
                too_close_to_player,
                jnp.logical_not(
                    is_position_in_bounds_not_in_mob_not_colliding(
                        state, proposed_position, collision_map
                    )
                ),
            ),
        )

        is_attacking_player = jnp.logical_and(
            is_attacking_player,
            ranged_mobs.attack_cooldown[state.player_level, ranged_mob_index] <= 0,
        )
        is_attacking_player = jnp.logical_and(
            is_attacking_player, ranged_mobs.mask[state.player_level, ranged_mob_index]
        )

        # Spawn projectile
        can_spawn_projectile = (
            state.mob_projectiles.mask[state.player_level].sum()
            < static_params.max_mob_projectiles
        )
        new_projectile_position = ranged_mobs.position[
            state.player_level, ranged_mob_index
        ]

        is_spawning_projectile = jnp.logical_and(
            is_attacking_player, can_spawn_projectile
        )

        new_mob_projectiles, new_mob_projectile_directions = spawn_projectile(
            state,
            static_params,
            state.mob_projectiles,
            state.mob_projectile_directions,
            new_projectile_position,
            is_spawning_projectile,
            player_move_direction,
            RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_MAPPING[
                ranged_mobs.type_id[state.player_level, ranged_mob_index]
            ],
        )

        state = state.replace(
            mob_projectiles=new_mob_projectiles,
            mob_projectile_directions=new_mob_projectile_directions,
        )

        proposed_position = jax.lax.select(
            is_attacking_player,
            ranged_mobs.position[state.player_level, ranged_mob_index],
            proposed_position,
        )

        new_cooldown = jax.lax.select(
            is_attacking_player,
            4,
            ranged_mobs.attack_cooldown[state.player_level, ranged_mob_index] - 1,
        )

        valid_move = is_position_in_bounds_not_in_mob_not_colliding(
            state, proposed_position, collision_map
        )

        position = jax.lax.select(
            valid_move,
            proposed_position,
            ranged_mobs.position[state.player_level, ranged_mob_index],
        )

        should_not_despawn = (
            jnp.abs(
                ranged_mobs.position[state.player_level, ranged_mob_index]
                - state.player_position
            ).sum()
            < params.mob_despawn_distance
        )
        should_not_despawn = jnp.logical_or(
            should_not_despawn, is_fighting_boss(state, static_params)
        )

        # Clear our old entry if we are alive
        new_mob_map = state.mob_map.at[
            state.player_level,
            state.ranged_mobs.position[state.player_level, ranged_mob_index, 0],
            state.ranged_mobs.position[state.player_level, ranged_mob_index, 1],
        ].set(
            jnp.logical_and(
                state.mob_map[
                    state.player_level,
                    state.ranged_mobs.position[state.player_level, ranged_mob_index, 0],
                    state.ranged_mobs.position[state.player_level, ranged_mob_index, 1],
                ],
                jnp.logical_not(ranged_mobs.mask[state.player_level, ranged_mob_index]),
            )
        )
        new_mask = jnp.logical_and(
            state.ranged_mobs.mask[state.player_level, ranged_mob_index],
            should_not_despawn,
        )
        # Enter new entry if we are alive and not despawning this timestep
        new_mob_map = new_mob_map.at[state.player_level, position[0], position[1]].set(
            jnp.logical_or(
                new_mob_map[state.player_level, position[0], position[1]], new_mask
            )
        )

        state = state.replace(
            ranged_mobs=state.ranged_mobs.replace(
                position=state.ranged_mobs.position.at[
                    state.player_level, ranged_mob_index
                ].set(position),
                attack_cooldown=state.ranged_mobs.attack_cooldown.at[
                    state.player_level, ranged_mob_index
                ].set(new_cooldown),
                mask=state.ranged_mobs.mask.at[
                    state.player_level, ranged_mob_index
                ].set(
                    jnp.logical_and(
                        state.ranged_mobs.mask[state.player_level, ranged_mob_index],
                        should_not_despawn,
                    )
                ),
            ),
            mob_map=new_mob_map,
        )

        return (rng, state), None

    rng, _rng = jax.random.split(rng)
    (rng, state), _ = jax.lax.scan(
        _move_ranged_mob, (rng, state), jnp.arange(static_params.max_ranged_mobs)
    )

    # Move projectiles
    def _move_mob_projectile(rng_and_state, projectile_index):
        rng, state = rng_and_state
        projectiles = state.mob_projectiles

        proposed_position = (
            projectiles.position[state.player_level, projectile_index]
            + state.mob_projectile_directions[state.player_level, projectile_index]
        )

        proposed_position_in_player = (proposed_position == state.player_position).all()

        proposed_position_in_bounds = in_bounds(state, proposed_position)
        in_wall = is_in_solid_block(state, proposed_position)
        in_wall = jnp.logical_and(
            in_wall,
            jnp.logical_not(
                state.map[state.player_level][
                    proposed_position[0], proposed_position[1]
                ]
                == BlockType.WATER.value
            ),
        )  # Arrows can go over water
        in_mob = is_in_mob(state, proposed_position)

        continue_move = jnp.logical_and(
            proposed_position_in_bounds, jnp.logical_not(in_wall)
        )
        continue_move = jnp.logical_and(continue_move, jnp.logical_not(in_mob))

        hit_player0 = jnp.logical_and(
            (
                projectiles.position[state.player_level, projectile_index]
                == state.player_position
            ).all(),
            projectiles.mask[state.player_level, projectile_index],
        )

        hit_player1 = jnp.logical_and(
            proposed_position_in_player,
            projectiles.mask[state.player_level, projectile_index],
        )
        hit_player = jnp.logical_or(hit_player0, hit_player1)

        continue_move = jnp.logical_and(continue_move, jnp.logical_not(hit_player))

        position = proposed_position

        # Clear our old entry if we are alive
        new_mask = jnp.logical_and(
            continue_move, projectiles.mask[state.player_level, projectile_index]
        )

        hit_bench_or_furnace = jnp.logical_or(
            state.map[state.player_level, position[0], position[1]]
            == BlockType.FURNACE.value,
            state.map[state.player_level, position[0], position[1]]
            == BlockType.CRAFTING_TABLE.value,
        )
        removing_block = jnp.logical_and(
            hit_bench_or_furnace, projectiles.mask[state.player_level, projectile_index]
        )

        new_block = jax.lax.select(
            removing_block,
            BlockType.PATH.value,
            state.map[state.player_level, position[0], position[1]],
        )

        projectile_type = state.mob_projectiles.type_id[
            state.player_level, projectile_index
        ]
        projectile_damage = get_damage_done_to_player(
            state,
            static_params,
            MOB_TYPE_DAMAGE_MAPPING[projectile_type, MobType.PROJECTILE.value],
        )

        state = state.replace(
            mob_projectiles=state.mob_projectiles.replace(
                position=state.mob_projectiles.position.at[
                    state.player_level, projectile_index
                ].set(position),
                mask=state.mob_projectiles.mask.at[
                    state.player_level, projectile_index
                ].set(new_mask),
            ),
            player_health=state.player_health - projectile_damage * hit_player,
            is_sleeping=jnp.logical_and(state.is_sleeping, jnp.logical_not(hit_player)),
            is_resting=jnp.logical_and(state.is_resting, jnp.logical_not(hit_player)),
            map=state.map.at[state.player_level, position[0], position[1]].set(
                new_block
            ),
        )

        return (rng, state), None

    rng, _rng = jax.random.split(rng)
    (rng, state), _ = jax.lax.scan(
        _move_mob_projectile,
        (rng, state),
        jnp.arange(static_params.max_mob_projectiles),
    )

    def _move_player_projectile(rng_and_state, projectile_index):
        rng, state = rng_and_state
        projectiles = state.player_projectiles

        projectile_type = state.player_projectiles.type_id[
            state.player_level, projectile_index
        ]

        projectile_damage_vector = (
            MOB_TYPE_DAMAGE_MAPPING[projectile_type, MobType.PROJECTILE.value]
            * projectiles.mask[state.player_level, projectile_index]
        )

        is_arrow = jnp.logical_or(
            projectile_type == ProjectileType.ARROW.value,
            projectile_type == ProjectileType.ARROW2.value,
        )

        # Bow enchantment
        arrow_damage_add = jnp.zeros(3, dtype=jnp.float32)
        arrow_damage_add = arrow_damage_add.at[state.bow_enchantment].set(
            projectile_damage_vector[0] / 2
        )
        arrow_damage_add = arrow_damage_add.at[0].set(0)

        projectile_damage_vector += jax.lax.select(
            is_arrow,
            arrow_damage_add,
            jnp.zeros(3, dtype=jnp.float32),
        )

        # Apply attribute scaling
        arrow_damage_coeff = 1 + 0.2 * (state.player_dexterity - 1)
        magic_damage_coeff = 1 + 0.5 * (state.player_intelligence - 1)

        projectile_damage_vector *= jax.lax.select(
            is_arrow,
            arrow_damage_coeff,
            1.0,
        )

        projectile_damage_vector *= jax.lax.select(
            jnp.logical_or(
                projectile_type == ProjectileType.FIREBALL.value,
                projectile_type == ProjectileType.ICEBALL.value,
            ),
            magic_damage_coeff,
            1.0,
        )

        proposed_position = (
            projectiles.position[state.player_level, projectile_index]
            + state.player_projectile_directions[state.player_level, projectile_index]
        )

        proposed_position_in_bounds = in_bounds(state, proposed_position)
        in_wall = is_in_solid_block(state, proposed_position)
        in_wall = jnp.logical_and(
            in_wall,
            jnp.logical_not(
                state.map[state.player_level][
                    proposed_position[0], proposed_position[1]
                ]
                == BlockType.WATER.value
            ),
        )  # Arrows can go over water

        state, did_attack_mob0, did_kill_mob0 = attack_mob(
            state,
            projectiles.position[state.player_level, projectile_index],
            projectile_damage_vector,
            False,
        )

        projectile_damage_vector = projectile_damage_vector * (1 - did_attack_mob0)

        state, did_attack_mob1, did_kill_mob1 = attack_mob(
            state, proposed_position, projectile_damage_vector, False
        )

        did_attack_mob = jnp.logical_or(did_attack_mob0, did_attack_mob1)

        continue_move = jnp.logical_and(
            proposed_position_in_bounds, jnp.logical_not(in_wall)
        )
        continue_move = jnp.logical_and(continue_move, jnp.logical_not(did_attack_mob))
        position = proposed_position

        # Clear our old entry if we are alive
        new_mask = jnp.logical_and(
            continue_move, projectiles.mask[state.player_level, projectile_index]
        )

        state = state.replace(
            player_projectiles=state.player_projectiles.replace(
                position=state.player_projectiles.position.at[
                    state.player_level, projectile_index
                ].set(position),
                mask=state.player_projectiles.mask.at[
                    state.player_level, projectile_index
                ].set(new_mask),
            ),
        )

        return (rng, state), None

    rng, _rng = jax.random.split(rng)
    (rng, state), _ = jax.lax.scan(
        _move_player_projectile,
        (rng, state),
        jnp.arange(static_params.max_player_projectiles),
    )

    return state


def update_player_intrinsics(state, action, static_params):
    # Start sleeping?
    is_starting_sleep = jnp.logical_and(
        action == Action.SLEEP.value, state.player_energy < get_max_energy(state)
    )
    new_is_sleeping = jnp.logical_or(state.is_sleeping, is_starting_sleep)
    state = state.replace(is_sleeping=new_is_sleeping)

    # Wake up?
    is_waking_up = jnp.logical_and(
        state.player_energy >= get_max_energy(state), state.is_sleeping
    )
    new_is_sleeping = jnp.logical_and(state.is_sleeping, jnp.logical_not(is_waking_up))
    state = state.replace(
        is_sleeping=new_is_sleeping,
        achievements=state.achievements.at[Achievement.WAKE_UP.value].set(
            jnp.logical_or(state.achievements[Achievement.WAKE_UP.value], is_waking_up)
        ),
    )

    # Start resting?
    is_starting_rest = jnp.logical_and(
        action == Action.REST.value, state.player_health < get_max_health(state)
    )
    new_is_resting = jnp.logical_or(state.is_resting, is_starting_rest)
    state = state.replace(is_resting=new_is_resting)

    # Wake up from resting
    is_waking_up = jnp.logical_and(
        state.is_resting,
        jnp.logical_or(
            state.player_health >= get_max_health(state),
            jnp.logical_or(state.player_food <= 0, state.player_drink <= 0),
        ),
    )
    new_is_resting = jnp.logical_and(state.is_resting, jnp.logical_not(is_waking_up))
    state = state.replace(
        is_resting=new_is_resting,
    )

    not_boss = jnp.logical_not(is_fighting_boss(state, static_params))

    intrinsic_decay_coeff = 1.0 - (0.125 * (state.player_dexterity - 1))

    # Hunger
    hunger_add = jax.lax.select(state.is_sleeping, 0.5, 1.0) * intrinsic_decay_coeff
    new_hunger = state.player_hunger + hunger_add

    hungered_food = jnp.maximum(state.player_food - 1 * not_boss, 0)
    new_food = jax.lax.select(new_hunger > 25, hungered_food, state.player_food)
    new_hunger = jax.lax.select(new_hunger > 25, 0.0, new_hunger)

    state = state.replace(
        player_hunger=new_hunger,
        player_food=new_food,
    )

    # Thirst
    thirst_add = jax.lax.select(state.is_sleeping, 0.5, 1.0) * intrinsic_decay_coeff
    new_thirst = state.player_thirst + thirst_add
    thirsted_drink = jnp.maximum(state.player_drink - 1 * not_boss, 0)
    new_drink = jax.lax.select(new_thirst > 20, thirsted_drink, state.player_drink)
    new_thirst = jax.lax.select(new_thirst > 20, 0.0, new_thirst)

    state = state.replace(
        player_thirst=new_thirst,
        player_drink=new_drink,
    )

    # Fatigue
    new_fatigue = jax.lax.select(
        state.is_sleeping,
        jnp.minimum(state.player_fatigue - 1, 0),
        state.player_fatigue + intrinsic_decay_coeff,
    )

    new_energy = jax.lax.select(
        new_fatigue > 30,
        jnp.maximum(state.player_energy - 1 * not_boss, 0),
        state.player_energy,
    )
    new_fatigue = jax.lax.select(new_fatigue > 30, 0.0, new_fatigue)

    new_energy = jax.lax.select(
        new_fatigue < -10,
        jnp.minimum(state.player_energy + 1, get_max_energy(state)),
        new_energy,
    )
    new_fatigue = jax.lax.select(new_fatigue < -10, 0.0, new_fatigue)

    state = state.replace(
        player_fatigue=new_fatigue,
        player_energy=new_energy,
    )

    # Health
    necessities = jnp.array(
        [
            state.player_food > 0,
            state.player_drink > 0,
            jnp.logical_or(state.player_energy > 0, state.is_sleeping),
        ],
        dtype=bool,
    )

    all_necessities = necessities.all()
    recover_all = jax.lax.select(state.is_sleeping, 2.0, 1.0)
    recover_not_all = jax.lax.select(state.is_sleeping, -0.5, -1.0) * not_boss
    recover_add = jax.lax.select(all_necessities, recover_all, recover_not_all)

    new_recover = state.player_recover + recover_add

    recovered_health = jnp.minimum(state.player_health + 1, get_max_health(state))
    derecovered_health = state.player_health - 1

    new_health = jax.lax.select(new_recover > 25, recovered_health, state.player_health)
    new_recover = jax.lax.select(new_recover > 25, 0.0, new_recover)
    new_health = jax.lax.select(new_recover < -15, derecovered_health, new_health)
    new_recover = jax.lax.select(new_recover < -15, 0.0, new_recover)

    state = state.replace(
        player_recover=new_recover,
        player_health=new_health,
    )

    # Mana
    mana_recover_coeff = 1 + 0.25 * (state.player_intelligence - 1)
    new_recover_mana = (
        jax.lax.select(
            state.is_sleeping,
            state.player_recover_mana + 2,
            state.player_recover_mana + 1,
        )
        * mana_recover_coeff
    )

    new_mana = jax.lax.select(
        new_recover_mana > 30, state.player_mana + 1, state.player_mana
    )
    new_recover_mana = jax.lax.select(new_recover_mana > 30, 0.0, new_recover_mana)

    state = state.replace(
        player_recover_mana=new_recover_mana,
        player_mana=new_mana,
    )

    return state


def update_plants(state, static_params):
    growing_plants_age = state.growing_plants_age + 1
    growing_plants_age *= state.growing_plants_mask

    finished_growing_plants = growing_plants_age >= 600

    new_plant_blocks = jnp.where(
        finished_growing_plants,
        BlockType.RIPE_PLANT.value,
        BlockType.PLANT.value,
    )

    def _set_plant_block(map, plant_index):
        new_block = jax.lax.select(
            finished_growing_plants[plant_index],
            new_plant_blocks[plant_index],
            map[
                state.growing_plants_positions[plant_index][0],
                state.growing_plants_positions[plant_index][1],
            ],
        )
        map = map.at[
            state.growing_plants_positions[plant_index][0],
            state.growing_plants_positions[plant_index][1],
        ].set(new_block)
        return map, None

    new_map, _ = jax.lax.scan(
        _set_plant_block,
        state.map[0],
        jnp.arange(static_params.max_growing_plants),
    )

    new_whole_map = state.map.at[0].set(new_map)

    state = state.replace(
        map=new_whole_map,
        growing_plants_age=growing_plants_age,
    )

    return state


def move_player(state, action, params):
    proposed_position = state.player_position + DIRECTIONS[action]

    valid_move = is_position_in_bounds_not_in_mob_not_colliding(
        state, proposed_position, COLLISION_LAND_CREATURE
    )
    valid_move = jnp.logical_or(valid_move, params.god_mode)

    position = state.player_position + valid_move.astype(jnp.int32) * DIRECTIONS[action]

    is_new_direction = jnp.sum(jnp.abs(DIRECTIONS[action])) != 0
    new_direction = (
        state.player_direction * (1 - is_new_direction) + action * is_new_direction
    )

    state = state.replace(
        player_position=position,
        player_direction=new_direction,
    )

    return state


def spawn_mobs(state, rng, params, static_params):
    player_distance_map = get_distance_map(
        state.player_position, static_params.map_size
    )
    grave_map = jnp.logical_or(
        state.map[state.player_level] == BlockType.GRAVE.value,
        jnp.logical_or(
            state.map[state.player_level] == BlockType.GRAVE2.value,
            state.map[state.player_level] == BlockType.GRAVE3.value,
        ),
    )

    monster_spawn_coeff = (
        1
        + (state.monsters_killed[state.player_level] < MONSTERS_KILLED_TO_CLEAR_LEVEL)
        * 2
    )  # Triple spawn rate if we are on an uncleared level

    monster_spawn_coeff *= jax.lax.select(
        is_fighting_boss(state, static_params),
        is_boss_spawn_wave(state, static_params) * 1000,
        1,
    )

    # Passive mobs
    can_spawn_passive_mob = (
        state.passive_mobs.mask[state.player_level].sum()
        < static_params.max_passive_mobs
    )

    rng, _rng = jax.random.split(rng)
    can_spawn_passive_mob = jnp.logical_and(
        can_spawn_passive_mob,
        jax.random.uniform(_rng) < FLOOR_MOB_SPAWN_CHANCE[state.player_level, 0],
    )

    can_spawn_passive_mob = jnp.logical_and(
        can_spawn_passive_mob, jnp.logical_not(is_fighting_boss(state, static_params))
    )

    all_valid_blocks_map = jnp.logical_or(
        state.map[state.player_level] == BlockType.GRASS.value,
        jnp.logical_or(
            state.map[state.player_level] == BlockType.PATH.value,
            jnp.logical_or(
                state.map[state.player_level] == BlockType.FIRE_GRASS.value,
                state.map[state.player_level] == BlockType.ICE_GRASS.value,
            ),
        ),
    )
    grass_map = state.map[state.player_level] == BlockType.GRASS.value
    path_map = state.map[state.player_level] == BlockType.PATH.value
    new_passive_mob_type = FLOOR_MOB_MAPPING[state.player_level, MobType.PASSIVE.value]

    passive_mobs_can_spawn_map = all_valid_blocks_map

    passive_mobs_can_spawn_map = jnp.logical_and(
        passive_mobs_can_spawn_map, player_distance_map > 3
    )
    passive_mobs_can_spawn_map = jnp.logical_and(
        passive_mobs_can_spawn_map, player_distance_map < params.mob_despawn_distance
    )
    passive_mobs_can_spawn_map = jnp.logical_and(
        passive_mobs_can_spawn_map, jnp.logical_not(state.mob_map[state.player_level])
    )
    can_spawn_passive_mob = jnp.logical_and(
        can_spawn_passive_mob, passive_mobs_can_spawn_map.sum() > 0
    )

    rng, _rng = jax.random.split(rng)
    passive_mob_position = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        shape=(1,),
        p=jnp.reshape(passive_mobs_can_spawn_map, -1)
        / jnp.sum(passive_mobs_can_spawn_map),
    )
    passive_mob_position = jnp.array(
        [
            passive_mob_position // static_params.map_size[0],
            passive_mob_position % static_params.map_size[1],
        ]
    ).T.astype(jnp.int32)[0]

    new_passive_mob_index = jnp.argmax(
        jnp.logical_not(state.passive_mobs.mask[state.player_level])
    )

    new_passive_mob_position = jax.lax.select(
        can_spawn_passive_mob,
        passive_mob_position,
        state.passive_mobs.position[state.player_level, new_passive_mob_index],
    )

    new_passive_mob_health = jax.lax.select(
        can_spawn_passive_mob,
        MOB_TYPE_HEALTH_MAPPING[new_passive_mob_type, MobType.PASSIVE.value],
        state.passive_mobs.health[state.player_level, new_passive_mob_index],
    )

    new_passive_mob_mask = jax.lax.select(
        can_spawn_passive_mob,
        True,
        state.passive_mobs.mask[state.player_level, new_passive_mob_index],
    )

    passive_mobs = Mobs(
        position=state.passive_mobs.position.at[
            state.player_level, new_passive_mob_index
        ].set(new_passive_mob_position),
        health=state.passive_mobs.health.at[
            state.player_level, new_passive_mob_index
        ].set(new_passive_mob_health),
        mask=state.passive_mobs.mask.at[state.player_level, new_passive_mob_index].set(
            new_passive_mob_mask
        ),
        attack_cooldown=state.passive_mobs.attack_cooldown,
        type_id=state.passive_mobs.type_id.at[
            state.player_level, new_passive_mob_index
        ].set(new_passive_mob_type),
    )

    state = state.replace(
        passive_mobs=passive_mobs,
        mob_map=state.mob_map.at[
            state.player_level, new_passive_mob_position[0], new_passive_mob_position[1]
        ].set(
            jnp.logical_or(
                state.mob_map[
                    state.player_level,
                    new_passive_mob_position[0],
                    new_passive_mob_position[1],
                ],
                new_passive_mob_mask,
            )
        ),
    )

    # Monsters
    monsters_can_spawn_player_range_map = player_distance_map > 9
    monsters_can_spawn_player_range_map_boss = player_distance_map <= 6

    monsters_can_spawn_player_range_map = jax.lax.select(
        is_fighting_boss(state, static_params),
        monsters_can_spawn_player_range_map_boss,
        monsters_can_spawn_player_range_map,
    )

    # Melee mobs
    can_spawn_melee_mob = (
        state.melee_mobs.mask[state.player_level].sum() < static_params.max_melee_mobs
    )

    new_melee_mob_type = FLOOR_MOB_MAPPING[state.player_level, MobType.MELEE.value]
    new_melee_mob_type_boss = FLOOR_MOB_MAPPING[
        state.boss_progress, MobType.MELEE.value
    ]

    new_melee_mob_type = jax.lax.select(
        is_fighting_boss(state, static_params),
        new_melee_mob_type_boss,
        new_melee_mob_type,
    )

    rng, _rng = jax.random.split(rng)
    melee_mob_spawn_chance = FLOOR_MOB_SPAWN_CHANCE[
        state.player_level, 1
    ] + FLOOR_MOB_SPAWN_CHANCE[state.player_level, 3] * jnp.square(
        1 - state.light_level
    )
    can_spawn_melee_mob = jnp.logical_and(
        can_spawn_melee_mob,
        jax.random.uniform(_rng) < melee_mob_spawn_chance * monster_spawn_coeff,
    )

    melee_mobs_can_spawn_map = jax.lax.select(
        is_fighting_boss(state, static_params), grave_map, all_valid_blocks_map
    )

    melee_mobs_can_spawn_map = jnp.logical_and(
        melee_mobs_can_spawn_map, monsters_can_spawn_player_range_map
    )
    melee_mobs_can_spawn_map = jnp.logical_and(
        melee_mobs_can_spawn_map, player_distance_map < params.mob_despawn_distance
    )
    melee_mobs_can_spawn_map = jnp.logical_and(
        melee_mobs_can_spawn_map, jnp.logical_not(state.mob_map[state.player_level])
    )

    can_spawn_melee_mob = jnp.logical_and(
        can_spawn_melee_mob, melee_mobs_can_spawn_map.sum() > 0
    )

    rng, _rng = jax.random.split(rng)
    melee_mob_position = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        shape=(1,),
        p=jnp.reshape(melee_mobs_can_spawn_map, -1) / jnp.sum(melee_mobs_can_spawn_map),
    )
    melee_mob_position = jnp.array(
        [
            melee_mob_position // static_params.map_size[0],
            melee_mob_position % static_params.map_size[1],
        ]
    ).T.astype(jnp.int32)[0]

    new_melee_mob_index = jnp.argmax(
        jnp.logical_not(state.melee_mobs.mask[state.player_level])
    )

    new_melee_mob_position = jax.lax.select(
        can_spawn_melee_mob,
        melee_mob_position,
        state.melee_mobs.position[state.player_level, new_melee_mob_index],
    )

    new_melee_mob_health = jax.lax.select(
        can_spawn_melee_mob,
        MOB_TYPE_HEALTH_MAPPING[new_melee_mob_type, MobType.MELEE.value],
        state.melee_mobs.health[state.player_level, new_melee_mob_index],
    )

    new_melee_mob_mask = jax.lax.select(
        can_spawn_melee_mob,
        True,
        state.melee_mobs.mask[state.player_level, new_melee_mob_index],
    )

    melee_mobs = Mobs(
        position=state.melee_mobs.position.at[
            state.player_level, new_melee_mob_index
        ].set(new_melee_mob_position),
        health=state.melee_mobs.health.at[state.player_level, new_melee_mob_index].set(
            new_melee_mob_health
        ),
        mask=state.melee_mobs.mask.at[state.player_level, new_melee_mob_index].set(
            new_melee_mob_mask
        ),
        attack_cooldown=state.melee_mobs.attack_cooldown,
        type_id=state.melee_mobs.type_id.at[
            state.player_level, new_melee_mob_index
        ].set(new_melee_mob_type),
    )

    state = state.replace(
        melee_mobs=melee_mobs,
        mob_map=state.mob_map.at[
            state.player_level, new_melee_mob_position[0], new_melee_mob_position[1]
        ].set(
            jnp.logical_or(
                state.mob_map[
                    state.player_level,
                    new_melee_mob_position[0],
                    new_melee_mob_position[1],
                ],
                new_melee_mob_mask,
            )
        ),
    )

    # Ranged mobs
    can_spawn_ranged_mob = (
        state.ranged_mobs.mask[state.player_level].sum() < static_params.max_ranged_mobs
    )

    new_ranged_mob_type = FLOOR_MOB_MAPPING[state.player_level, MobType.RANGED.value]
    new_ranged_mob_type_boss = FLOOR_MOB_MAPPING[
        state.boss_progress, MobType.RANGED.value
    ]

    new_ranged_mob_type = jax.lax.select(
        is_fighting_boss(state, static_params),
        new_ranged_mob_type_boss,
        new_ranged_mob_type,
    )

    rng, _rng = jax.random.split(rng)
    can_spawn_ranged_mob = jnp.logical_and(
        can_spawn_ranged_mob,
        jax.random.uniform(_rng)
        < FLOOR_MOB_SPAWN_CHANCE[state.player_level, 2] * monster_spawn_coeff,
    )

    # Hack for deep thing
    ranged_mobs_can_spawn_map = jax.lax.select(
        new_ranged_mob_type == 5,
        state.map[state.player_level] == BlockType.WATER.value,
        all_valid_blocks_map,
    )
    ranged_mobs_can_spawn_map = jax.lax.select(
        is_fighting_boss(state, static_params), grave_map, ranged_mobs_can_spawn_map
    )

    ranged_mobs_can_spawn_map = jnp.logical_and(
        ranged_mobs_can_spawn_map, monsters_can_spawn_player_range_map
    )
    ranged_mobs_can_spawn_map = jnp.logical_and(
        ranged_mobs_can_spawn_map, player_distance_map < params.mob_despawn_distance
    )
    ranged_mobs_can_spawn_map = jnp.logical_and(
        ranged_mobs_can_spawn_map, jnp.logical_not(state.mob_map[state.player_level])
    )

    can_spawn_ranged_mob = jnp.logical_and(
        can_spawn_ranged_mob, ranged_mobs_can_spawn_map.sum() > 0
    )

    rng, _rng = jax.random.split(rng)
    ranged_mob_position = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        shape=(1,),
        p=jnp.reshape(ranged_mobs_can_spawn_map, -1)
        / jnp.sum(ranged_mobs_can_spawn_map),
    )
    ranged_mob_position = jnp.array(
        [
            ranged_mob_position // static_params.map_size[0],
            ranged_mob_position % static_params.map_size[1],
        ]
    ).T.astype(jnp.int32)[0]

    new_ranged_mob_index = jnp.argmax(
        jnp.logical_not(state.ranged_mobs.mask[state.player_level])
    )

    new_ranged_mob_position = jax.lax.select(
        can_spawn_ranged_mob,
        ranged_mob_position,
        state.ranged_mobs.position[state.player_level, new_ranged_mob_index],
    )

    new_ranged_mob_health = jax.lax.select(
        can_spawn_ranged_mob,
        MOB_TYPE_HEALTH_MAPPING[new_ranged_mob_type, MobType.RANGED.value],
        state.ranged_mobs.health[state.player_level, new_ranged_mob_index],
    )

    new_ranged_mob_mask = jax.lax.select(
        can_spawn_ranged_mob,
        True,
        state.ranged_mobs.mask[state.player_level, new_ranged_mob_index],
    )

    ranged_mobs = Mobs(
        position=state.ranged_mobs.position.at[
            state.player_level, new_ranged_mob_index
        ].set(new_ranged_mob_position),
        health=state.ranged_mobs.health.at[
            state.player_level, new_ranged_mob_index
        ].set(new_ranged_mob_health),
        mask=state.ranged_mobs.mask.at[state.player_level, new_ranged_mob_index].set(
            new_ranged_mob_mask
        ),
        attack_cooldown=state.ranged_mobs.attack_cooldown,
        type_id=state.ranged_mobs.type_id.at[
            state.player_level, new_ranged_mob_index
        ].set(new_ranged_mob_type),
    )

    state = state.replace(
        ranged_mobs=ranged_mobs,
        mob_map=state.mob_map.at[
            state.player_level, new_ranged_mob_position[0], new_ranged_mob_position[1]
        ].set(
            jnp.logical_or(
                state.mob_map[
                    state.player_level,
                    new_ranged_mob_position[0],
                    new_ranged_mob_position[1],
                ],
                new_ranged_mob_mask,
            )
        ),
    )

    return state


def change_floor(
    state: EnvState, action, env_params: EnvParams, static_params: StaticEnvParams
):
    on_down_ladder = (
        state.item_map[
            state.player_level, state.player_position[0], state.player_position[1]
        ]
        == ItemType.LADDER_DOWN.value
    )
    is_moving_down = jnp.logical_and(
        action == Action.DESCEND.value,
        jnp.logical_or(
            env_params.god_mode,
            jnp.logical_and(
                on_down_ladder,
                state.monsters_killed[state.player_level]
                >= MONSTERS_KILLED_TO_CLEAR_LEVEL,
            ),
        ),
    )
    is_moving_down = jnp.logical_and(
        is_moving_down, state.player_level < static_params.num_levels - 1
    )

    moving_down_position = state.up_ladders[state.player_level + 1]

    on_up_ladder = (
        state.item_map[
            state.player_level, state.player_position[0], state.player_position[1]
        ]
        == ItemType.LADDER_UP.value
    )
    is_moving_up = jnp.logical_and(
        action == Action.ASCEND.value,
        jnp.logical_or(
            env_params.god_mode,
            on_up_ladder,
        ),
    )
    is_moving_up = jnp.logical_and(is_moving_up, state.player_level > 0)

    moving_up_position = state.down_ladders[state.player_level - 1]

    is_not_moving = jnp.logical_not(jnp.logical_or(is_moving_up, is_moving_down))

    delta_floor = 1 * is_moving_down - 1 * is_moving_up
    position = (
        (state.player_position * is_not_moving)
        + (is_moving_down * moving_down_position)
        + (is_moving_up * moving_up_position)
    )

    move_down_achievement = LEVEL_ACHIEVEMENT_MAP[state.player_level + delta_floor]

    new_achievements = state.achievements.at[move_down_achievement].set(
        jnp.logical_or(
            (state.player_level + delta_floor) != 0,
            state.achievements[move_down_achievement],
        )
    )

    new_floor = jnp.logical_and(
        (state.player_level + delta_floor) != 0,
        jnp.logical_not(state.achievements[move_down_achievement]),
    )

    state = state.replace(
        player_level=state.player_level + delta_floor,
        player_position=position,
        achievements=new_achievements,
        player_xp=state.player_xp + 1 * new_floor,
    )

    return state


def shoot_projectile(state: EnvState, action: int, static_params: StaticEnvParams):
    # Arrow
    is_shooting_arrow = jnp.logical_and(
        action == Action.SHOOT_ARROW.value,
        jnp.logical_and(
            state.inventory.bow >= 1,
            jnp.logical_and(
                state.inventory.arrows >= 1,
                state.player_projectiles.mask[state.player_level].sum()
                < static_params.max_player_projectiles,
            ),
        ),
    )

    new_player_projectiles, new_player_projectile_directions = spawn_projectile(
        state,
        static_params,
        state.player_projectiles,
        state.player_projectile_directions,
        state.player_position,
        is_shooting_arrow,
        DIRECTIONS[state.player_direction],
        ProjectileType.ARROW2.value,
    )

    new_achievements = state.achievements.at[Achievement.FIRE_BOW.value].set(
        jnp.logical_or(
            state.achievements[Achievement.FIRE_BOW.value], is_shooting_arrow
        )
    )

    return state.replace(
        player_projectiles=new_player_projectiles,
        player_projectile_directions=new_player_projectile_directions,
        inventory=state.inventory.replace(
            arrows=state.inventory.arrows - 1 * is_shooting_arrow
        ),
        achievements=new_achievements,
    )


def cast_spell(state, action, static_params):
    # Arrow
    is_casting_fireball = jnp.logical_and(
        action == Action.CAST_FIREBALL.value,
        jnp.logical_and(
            state.player_mana >= 2,
            state.player_projectiles.mask[state.player_level].sum()
            < static_params.max_player_projectiles,
        ),
    )
    is_casting_fireball = jnp.logical_and(is_casting_fireball, state.learned_spells[0])

    is_casting_iceball = jnp.logical_and(
        action == Action.CAST_ICEBALL.value,
        jnp.logical_and(
            state.player_mana >= 2,
            state.player_projectiles.mask[state.player_level].sum()
            < static_params.max_player_projectiles,
        ),
    )
    is_casting_iceball = jnp.logical_and(is_casting_iceball, state.learned_spells[1])

    is_casting_spell = jnp.logical_or(is_casting_fireball, is_casting_iceball)
    projectile_type = (
        is_casting_fireball * ProjectileType.FIREBALL.value
        + is_casting_iceball * ProjectileType.ICEBALL.value
    )

    new_player_projectiles, new_player_projectile_directions = spawn_projectile(
        state,
        static_params,
        state.player_projectiles,
        state.player_projectile_directions,
        state.player_position,
        is_casting_spell,
        DIRECTIONS[state.player_direction],
        projectile_type,
    )

    casting_achievement = (
        is_casting_fireball * Achievement.CAST_FIREBALL.value
        + is_casting_iceball * Achievement.CAST_ICEBALL.value
    )
    new_achievements = state.achievements.at[casting_achievement].set(
        jnp.logical_or(state.achievements[casting_achievement], is_casting_spell)
    )

    return state.replace(
        player_projectiles=new_player_projectiles,
        player_projectile_directions=new_player_projectile_directions,
        player_mana=state.player_mana - is_casting_spell * 2,
        achievements=new_achievements,
    )


def drink_potion(state, action):
    drinking_potion_index = -1
    is_drinking_potion = False

    # Red
    is_drinking_red_potion = jnp.logical_and(
        action == Action.DRINK_POTION_RED.value, state.inventory.potions[0] > 0
    )
    drinking_potion_index = (
        is_drinking_red_potion * 0
        + (1 - is_drinking_red_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_red_potion)

    # Green
    is_drinking_green_potion = jnp.logical_and(
        action == Action.DRINK_POTION_GREEN.value, state.inventory.potions[1] > 0
    )
    drinking_potion_index = (
        is_drinking_green_potion * 1
        + (1 - is_drinking_green_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_green_potion)

    # Blue
    is_drinking_blue_potion = jnp.logical_and(
        action == Action.DRINK_POTION_BLUE.value, state.inventory.potions[2] > 0
    )
    drinking_potion_index = (
        is_drinking_blue_potion * 2
        + (1 - is_drinking_blue_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_blue_potion)

    # Pink
    is_drinking_pink_potion = jnp.logical_and(
        action == Action.DRINK_POTION_PINK.value, state.inventory.potions[3] > 0
    )
    drinking_potion_index = (
        is_drinking_pink_potion * 3
        + (1 - is_drinking_pink_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_pink_potion)

    # Cyan
    is_drinking_cyan_potion = jnp.logical_and(
        action == Action.DRINK_POTION_CYAN.value, state.inventory.potions[4] > 0
    )
    drinking_potion_index = (
        is_drinking_cyan_potion * 4
        + (1 - is_drinking_cyan_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_cyan_potion)

    # Yellow
    is_drinking_yellow_potion = jnp.logical_and(
        action == Action.DRINK_POTION_YELLOW.value, state.inventory.potions[5] > 0
    )
    drinking_potion_index = (
        is_drinking_yellow_potion * 5
        + (1 - is_drinking_yellow_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_yellow_potion)

    # Potion mapping
    potion_effect_index = state.potion_mapping[drinking_potion_index]

    # Potion effect
    delta_health = 0
    delta_health += is_drinking_potion * (potion_effect_index == 0) * 8
    delta_health += is_drinking_potion * (potion_effect_index == 1) * (-3)

    delta_mana = 0
    delta_mana += is_drinking_potion * (potion_effect_index == 2) * 8
    delta_mana += is_drinking_potion * (potion_effect_index == 3) * (-3)

    delta_energy = 0
    delta_energy += is_drinking_potion * (potion_effect_index == 4) * 8
    delta_energy += is_drinking_potion * (potion_effect_index == 5) * (-3)

    new_achievements = state.achievements.at[Achievement.DRINK_POTION.value].set(
        jnp.logical_or(
            state.achievements[Achievement.DRINK_POTION.value], is_drinking_potion
        )
    )

    return state.replace(
        inventory=state.inventory.replace(
            potions=state.inventory.potions.at[drinking_potion_index].set(
                state.inventory.potions[drinking_potion_index] - 1 * is_drinking_potion
            )
        ),
        player_health=state.player_health + delta_health,
        player_mana=state.player_mana + delta_mana,
        player_energy=state.player_energy + delta_energy,
        achievements=new_achievements,
    )


def read_book(rng, state, action):
    is_reading_book = jnp.logical_and(
        action == Action.READ_BOOK.value, state.inventory.books > 0
    )
    spells_to_learn = jnp.logical_not(state.learned_spells).astype(float)
    spells_to_learn /= spells_to_learn.sum()

    rng, _rng = jax.random.split(rng)
    spell_to_learn_index = jax.random.choice(
        _rng, jnp.arange(2), shape=(), p=spells_to_learn
    )

    learn_spell_achievement = jax.lax.select(
        spell_to_learn_index,
        Achievement.LEARN_ICEBALL.value,
        Achievement.LEARN_FIREBALL.value,
    )

    new_achievements = state.achievements.at[learn_spell_achievement].set(
        jnp.logical_or(state.achievements[learn_spell_achievement], is_reading_book)
    )

    return state.replace(
        inventory=state.inventory.replace(
            books=state.inventory.books - 1 * is_reading_book
        ),
        learned_spells=state.learned_spells.at[spell_to_learn_index].set(
            jnp.logical_or(state.learned_spells[spell_to_learn_index], is_reading_book)
        ),
        achievements=new_achievements,
    )


def enchant(rng, state: EnvState, action):
    target_block_position = state.player_position + DIRECTIONS[state.player_direction]
    target_block = state.map[
        state.player_level, target_block_position[0], target_block_position[1]
    ]
    target_block_is_enchantment_table = jnp.logical_or(
        target_block == BlockType.ENCHANTMENT_TABLE_FIRE.value,
        target_block == BlockType.ENCHANTMENT_TABLE_ICE.value,
    )

    enchantment_type = jax.lax.select(
        target_block == BlockType.ENCHANTMENT_TABLE_FIRE.value, 1, 2
    )

    num_gems = jax.lax.select(
        target_block == BlockType.ENCHANTMENT_TABLE_FIRE.value,
        state.inventory.ruby,
        state.inventory.sapphire,
    )

    could_enchant = jnp.logical_and(
        state.player_mana >= 9,
        jnp.logical_and(target_block_is_enchantment_table, num_gems >= 1),
    )

    is_enchanting_bow = jnp.logical_and(
        could_enchant,
        jnp.logical_and(action == Action.ENCHANT_BOW.value, state.inventory.bow > 0),
    )

    is_enchanting_sword = jnp.logical_and(
        could_enchant,
        jnp.logical_and(
            action == Action.ENCHANT_SWORD.value, state.inventory.sword > 0
        ),
    )

    is_enchanting_armour = jnp.logical_and(
        could_enchant,
        jnp.logical_and(
            action == Action.ENCHANT_ARMOUR.value, state.inventory.armour.sum() > 0
        ),
    )

    rng, _rng = jax.random.split(rng)
    unenchanted_armour = state.armour_enchantments == 0
    opposite_enchanted_armour = jnp.logical_and(
        state.armour_enchantments != 0, state.armour_enchantments != enchantment_type
    )

    armour_targets = (
        unenchanted_armour + (unenchanted_armour.sum() == 0) * opposite_enchanted_armour
    )
    armour_target = jax.random.choice(_rng, jnp.arange(4), shape=(), p=armour_targets)

    is_enchanting = jnp.logical_or(
        is_enchanting_sword, jnp.logical_or(is_enchanting_bow, is_enchanting_armour)
    )

    new_sword_enchantment = (
        is_enchanting_sword * enchantment_type
        + (1 - is_enchanting_sword) * state.sword_enchantment
    )
    new_bow_enchantment = (
        is_enchanting_bow * enchantment_type
        + (1 - is_enchanting_bow) * state.bow_enchantment
    )

    new_armour_enchantments = state.armour_enchantments.at[armour_target].set(
        is_enchanting_armour * enchantment_type
        + (1 - is_enchanting_armour) * state.armour_enchantments[armour_target]
    )

    new_sapphire = state.inventory.sapphire - 1 * is_enchanting * (
        enchantment_type == 2
    )
    new_ruby = state.inventory.ruby - 1 * is_enchanting * (enchantment_type == 1)
    new_mana = state.player_mana - 9 * is_enchanting

    new_achievements = state.achievements.at[Achievement.ENCHANT_SWORD.value].set(
        jnp.logical_or(
            state.achievements[Achievement.ENCHANT_SWORD.value], is_enchanting_sword
        )
    )

    new_achievements = new_achievements.at[Achievement.ENCHANT_ARMOUR.value].set(
        jnp.logical_or(
            new_achievements[Achievement.ENCHANT_ARMOUR.value], is_enchanting_armour
        )
    )

    return state.replace(
        sword_enchantment=new_sword_enchantment,
        bow_enchantment=new_bow_enchantment,
        armour_enchantments=new_armour_enchantments,
        inventory=state.inventory.replace(
            sapphire=new_sapphire,
            ruby=new_ruby,
        ),
        player_mana=new_mana,
        achievements=new_achievements,
    )


def boss_logic(state, static_params):
    new_achievements = state.achievements.at[Achievement.DEFEAT_NECROMANCER.value].set(
        jnp.logical_or(
            state.achievements[Achievement.DEFEAT_NECROMANCER.value],
            has_beaten_boss(state, static_params),
        )
    )

    return state.replace(
        boss_timesteps_to_spawn_this_round=state.boss_timesteps_to_spawn_this_round
        - 1 * is_fighting_boss(state, static_params),
        achievements=new_achievements,
    )


def calculate_inventory_achievements(state):
    # Some achievements (e.g. make_diamond_pickaxe) can be achieved in multiple ways (finding in chest or crafting)
    # Rather than duplicating achievement code, we simply look in the inventory for these types of achievements
    # at the end of each timestep

    # Wood
    achievements = state.achievements.at[Achievement.COLLECT_WOOD.value].set(
        jnp.logical_or(
            state.achievements[Achievement.COLLECT_WOOD.value], state.inventory.wood > 0
        )
    )
    # Stone
    achievements = achievements.at[Achievement.COLLECT_STONE.value].set(
        jnp.logical_or(
            achievements[Achievement.COLLECT_STONE.value], state.inventory.stone > 0
        )
    )
    # Coal
    achievements = achievements.at[Achievement.COLLECT_COAL.value].set(
        jnp.logical_or(
            achievements[Achievement.COLLECT_COAL.value], state.inventory.coal > 0
        )
    )
    # Iron
    achievements = achievements.at[Achievement.COLLECT_IRON.value].set(
        jnp.logical_or(
            achievements[Achievement.COLLECT_IRON.value], state.inventory.iron > 0
        )
    )
    # Diamond
    achievements = achievements.at[Achievement.COLLECT_DIAMOND.value].set(
        jnp.logical_or(
            achievements[Achievement.COLLECT_DIAMOND.value], state.inventory.diamond > 0
        )
    )
    # Ruby
    achievements = achievements.at[Achievement.COLLECT_RUBY.value].set(
        jnp.logical_or(
            achievements[Achievement.COLLECT_RUBY.value], state.inventory.ruby > 0
        )
    )
    # Sapphire
    achievements = achievements.at[Achievement.COLLECT_SAPPHIRE.value].set(
        jnp.logical_or(
            achievements[Achievement.COLLECT_SAPPHIRE.value],
            state.inventory.sapphire > 0,
        )
    )
    # Sapling
    achievements = achievements.at[Achievement.COLLECT_SAPLING.value].set(
        jnp.logical_or(
            achievements[Achievement.COLLECT_SAPLING.value], state.inventory.sapling > 0
        )
    )
    # Bow
    achievements = achievements.at[Achievement.FIND_BOW.value].set(
        jnp.logical_or(
            achievements[Achievement.FIND_BOW.value], state.inventory.bow > 0
        )
    )
    # Arrow
    achievements = achievements.at[Achievement.MAKE_ARROW.value].set(
        jnp.logical_or(
            achievements[Achievement.MAKE_ARROW.value], state.inventory.arrows > 0
        )
    )
    # Torch
    achievements = achievements.at[Achievement.MAKE_TORCH.value].set(
        jnp.logical_or(
            achievements[Achievement.MAKE_TORCH.value], state.inventory.torches > 0
        )
    )

    # Pickaxe
    achievements = achievements.at[Achievement.MAKE_WOOD_PICKAXE.value].set(
        jnp.logical_or(
            achievements[Achievement.MAKE_WOOD_PICKAXE.value],
            state.inventory.pickaxe >= 1,
        )
    )
    achievements = achievements.at[Achievement.MAKE_STONE_PICKAXE.value].set(
        jnp.logical_or(
            achievements[Achievement.MAKE_STONE_PICKAXE.value],
            state.inventory.pickaxe >= 2,
        )
    )
    achievements = achievements.at[Achievement.MAKE_IRON_PICKAXE.value].set(
        jnp.logical_or(
            achievements[Achievement.MAKE_IRON_PICKAXE.value],
            state.inventory.pickaxe >= 3,
        )
    )
    achievements = achievements.at[Achievement.MAKE_DIAMOND_PICKAXE.value].set(
        jnp.logical_or(
            achievements[Achievement.MAKE_DIAMOND_PICKAXE.value],
            state.inventory.pickaxe >= 4,
        )
    )

    # Sword
    achievements = achievements.at[Achievement.MAKE_WOOD_SWORD.value].set(
        jnp.logical_or(
            achievements[Achievement.MAKE_WOOD_SWORD.value], state.inventory.sword >= 1
        )
    )
    achievements = achievements.at[Achievement.MAKE_STONE_SWORD.value].set(
        jnp.logical_or(
            achievements[Achievement.MAKE_STONE_SWORD.value], state.inventory.sword >= 2
        )
    )
    achievements = achievements.at[Achievement.MAKE_IRON_SWORD.value].set(
        jnp.logical_or(
            achievements[Achievement.MAKE_IRON_SWORD.value], state.inventory.sword >= 3
        )
    )
    achievements = achievements.at[Achievement.MAKE_DIAMOND_SWORD.value].set(
        jnp.logical_or(
            achievements[Achievement.MAKE_DIAMOND_SWORD.value],
            state.inventory.sword >= 4,
        )
    )

    return state.replace(achievements=achievements)


def level_up_attributes(state, action, params):
    can_level_up = state.player_xp >= 1

    is_levelling_up_dex = jnp.logical_and(
        can_level_up,
        jnp.logical_and(
            action == Action.LEVEL_UP_DEXTERITY.value,
            state.player_dexterity < params.max_attribute,
        ),
    )
    is_levelling_up_str = jnp.logical_and(
        can_level_up,
        jnp.logical_and(
            action == Action.LEVEL_UP_STRENGTH.value,
            state.player_strength < params.max_attribute,
        ),
    )
    is_levelling_up_int = jnp.logical_and(
        can_level_up,
        jnp.logical_and(
            action == Action.LEVEL_UP_INTELLIGENCE.value,
            state.player_intelligence < params.max_attribute,
        ),
    )
    is_levelling_up = jnp.logical_or(
        is_levelling_up_dex, jnp.logical_or(is_levelling_up_str, is_levelling_up_int)
    )

    return state.replace(
        player_dexterity=state.player_dexterity + 1 * is_levelling_up_dex,
        player_strength=state.player_strength + 1 * is_levelling_up_str,
        player_intelligence=state.player_intelligence + 1 * is_levelling_up_int,
        player_xp=state.player_xp - 1 * is_levelling_up,
    )


def craftax_step(rng, state, action, params, static_params):
    init_achievements = state.achievements
    init_health = state.player_health

    # Interrupt action if sleeping or resting
    action = jax.lax.select(state.is_sleeping, Action.NOOP.value, action)
    action = jax.lax.select(state.is_resting, Action.NOOP.value, action)

    # Change floor
    state = change_floor(state, action, params, static_params)

    # Crafting
    state = do_crafting(state, action)

    # Interact (mining, melee attacking, eating plants, drinking water)
    rng, _rng = jax.random.split(rng)
    state = do_action(_rng, state, action, static_params)

    # Placing
    state = place_block(state, action, static_params)

    # Shooting
    state = shoot_projectile(state, action, static_params)

    # Casting
    state = cast_spell(state, action, static_params)

    # Potions
    state = drink_potion(state, action)

    # Read
    rng, _rng = jax.random.split(rng)
    state = read_book(_rng, state, action)

    # Enchant
    rng, _rng = jax.random.split(rng)
    state = enchant(_rng, state, action)

    # Boss
    state = boss_logic(state, static_params)

    # Attributes
    state = level_up_attributes(state, action, params)

    # Movement
    state = move_player(state, action, params)

    # Mobs
    rng, _rng = jax.random.split(rng)
    state = update_mobs(_rng, state, params, static_params)

    rng, _rng = jax.random.split(rng)
    state = spawn_mobs(state, _rng, params, static_params)

    # Plants
    state = update_plants(state, static_params)

    # Intrinsics
    state = update_player_intrinsics(state, action, static_params)

    # Cap inv
    state = clip_inventory_and_intrinsics(state, params)

    # Inventory achievements
    state = calculate_inventory_achievements(state)

    # Reward
    achievement_coefficients = ACHIEVEMENT_REWARD_MAP
    achievement_reward = (
        (state.achievements.astype(int) - init_achievements.astype(int))
        * achievement_coefficients
    ).sum()
    health_reward = (state.player_health - init_health) * 0.1
    reward = achievement_reward + health_reward

    rng, _rng = jax.random.split(rng)

    state = state.replace(
        timestep=state.timestep + 1,
        light_level=calculate_light_level(state.timestep + 1, params),
        state_rng=_rng,
    )

    return state, reward
