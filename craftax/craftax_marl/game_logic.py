from craftax_marl.util.game_logic_utils import *
from craftax_marl.util.maths_utils import *

def interplayer_interaction(state, block_position, is_doing_action, env_params, static_params):
    # If other player is down revive them, otherwise damage (if friendly fire is enabled)

    in_other_player = (jnp.expand_dims(state.player_position, axis=1) == jnp.expand_dims(block_position, axis=0)).all(axis=2).T
    player_interacting_with = jnp.argmax(in_other_player, axis=-1)

    is_interacting_with_other_player = jnp.logical_and(
        in_other_player.any(axis=-1),
        is_doing_action,
    )
    is_player_being_interacted_with = jnp.any(
        jnp.logical_and(
            jnp.arange(static_params.player_count)[:, None] == player_interacting_with,
            is_interacting_with_other_player[None, :]
        ),
        axis=-1
    )
    is_player_being_revived = jnp.logical_and(
        is_player_being_interacted_with,
        jnp.logical_not(state.player_alive),
    )

    damage_taken = jnp.zeros(static_params.player_count).at[player_interacting_with].add(
        is_interacting_with_other_player * get_damage_between_players(state, player_interacting_with)
    )
    damage_taken *= env_params.friendly_fire

    new_player_health = jnp.where(
        is_player_being_revived,
        1.0,
        state.player_health - damage_taken,
    )
    state = state.replace(
        player_health=new_player_health,
    )
    return state

def update_plants_with_eat(state, plant_position, is_eating_plant):
    is_plant = jax.vmap(
        jnp.equal, in_axes=(0, None)
    )(
        plant_position, 
        state.growing_plants_positions
    ).all(axis=-1)
    plant_index = jnp.argmax(
        is_plant, axis=-1
    )
    selected_plant_age = jnp.where(
        is_eating_plant,
        0,
        state.growing_plants_age[plant_index]
    )
    return state.growing_plants_age.at[plant_index].set(selected_plant_age)


def add_items_from_chest(rng, state, inventory, is_opening_chest):
    is_miner = state.player_specialization == Specialization.MINER.value
    is_warrior = state.player_specialization == Specialization.WARRIOR.value

    # Wood (60%)
    rng, _rng = jax.random.split(rng)
    is_looting_wood = jax.random.uniform(_rng) < 0.6 * is_opening_chest
    rng, _rng = jax.random.split(rng)
    wood_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=1, maxval=6) * is_looting_wood
    )

    # Torch (60%)
    rng, _rng = jax.random.split(rng)
    collect_prob = 0.1 + 0.5 * is_miner
    is_looting_torch = jax.random.uniform(_rng) < collect_prob * is_opening_chest
    rng, _rng = jax.random.split(rng)
    torch_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=4, maxval=8) * is_looting_torch
    )

    # Ores (60%)
    rng, _rng = jax.random.split(rng)
    is_looting_ore = jax.random.uniform(_rng) < collect_prob * is_opening_chest
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
    is_looting_potion = jax.random.uniform(_rng) < 0.5 * is_opening_chest
    rng, _rng = jax.random.split(rng)
    potion_loot_index = jax.random.randint(_rng, shape=(), minval=0, maxval=6)
    rng, _rng = jax.random.split(rng)
    potion_loot_amount = jax.random.randint(_rng, shape=(), minval=1, maxval=3)

    # Arrows (50%)
    rng, _rng = jax.random.split(rng)
    is_looting_arrows = jax.random.uniform(_rng) < 0.5 * is_opening_chest * is_warrior
    rng, _rng = jax.random.split(rng)
    arrows_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=4, maxval=9) * is_looting_arrows
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
    pickaxe_loot_level = jnp.where( # only miners can own pickaxes above level 1 (wood)
        is_miner,
        pickaxe_loot_level,
        1
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
            (jnp.arange(3) + 2).astype(int),
            shape=(),
            p=jnp.array([0.5, 0.3, 0.2]),
        )
        * is_looting_sword
    )
    sword_loot_level = jnp.where( # only warriors can own swords above level 2 (stone)
        is_warrior,
        sword_loot_level,
        2
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

    can_loot_book = jnp.logical_and(
        jnp.logical_not(state.chests_opened[state.player_level]),
        jnp.logical_or(state.player_level == 3, state.player_level == 4),
    )
    is_looting_book = jnp.logical_and(
        can_loot_book,
        is_opening_chest
    )

    # Update inventory
    return inventory.replace(
        wood=inventory.wood + wood_loot_amount,
        torches=inventory.torches + torch_loot_amount,
        coal=inventory.coal + coal_loot_amount,
        iron=inventory.iron + iron_loot_amount,
        diamond=inventory.diamond + diamond_loot_amount,
        sapphire=inventory.sapphire + sapphire_loot_amount,
        ruby=inventory.ruby + ruby_loot_amount,
        arrows=inventory.arrows + arrows_loot_amount,
        pickaxe=new_pickaxe_level,
        sword=new_sword_level,
        potions=inventory.potions.at[:, potion_loot_index].set(
            inventory.potions[:, potion_loot_index]
            + potion_loot_amount * is_looting_potion * is_opening_chest
        ),
        bow=new_bow_level,
        books=inventory.books + 1 * is_looting_book,
    )


def do_action(rng, state, action, env_params, static_params):
    is_forager = state.player_specialization == Specialization.FORAGER.value

    block_position = state.player_position + DIRECTIONS[state.player_direction]
    equal_block_placement = (jnp.expand_dims(block_position, axis=1) == jnp.expand_dims(block_position, axis=0)).all(axis=2)

    doing_action = jnp.logical_and(
        in_bounds(block_position, static_params),
        action == Action.DO.value
    )

    state, did_attack_mob, did_kill_mob = attack_mob(
        state, doing_action, block_position, get_player_damage_vector(state), is_forager
    )
    
    # Interact with other players (Damage/Revive)
    state = interplayer_interaction(state, block_position, doing_action, env_params, static_params)
    
    # BLOCKS
    # Tree
    can_mine_tree = True
    is_block_tree = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.TREE.value
    )
    is_block_fire_tree = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.FIRE_TREE.value
    )
    is_block_ice_shrub = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.ICE_SHRUB.value
    )

    is_block_tree_type = jnp.logical_or(
        is_block_tree, jnp.logical_or(is_block_fire_tree, is_block_ice_shrub)
    )
    is_mining_tree = jnp.logical_and(
        jnp.logical_and(
            is_block_tree_type,
            can_mine_tree,
        ),
        doing_action
    )
    tree_replacement_block = (
        is_block_tree * BlockType.GRASS.value
        + is_block_fire_tree * BlockType.FIRE_GRASS.value
        + is_block_ice_shrub * BlockType.ICE_GRASS.value
    )
    is_any_player_mining_tree = jnp.logical_and(
        equal_block_placement,
        is_mining_tree[:, None]
    ).any(axis=0)
    mined_tree_block = jnp.where(
        is_any_player_mining_tree,
        tree_replacement_block,
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]],
    )
    new_map = (
        state.map[state.player_level]
        .at[block_position[:, 0], block_position[:, 1]]
        .set(mined_tree_block)
    )
    new_inventory = state.inventory.replace(
        wood=state.inventory.wood + 1 * is_mining_tree
    )

    # Stone
    can_mine_stone = state.inventory.pickaxe >= 1
    is_block_stone = (
        state.map[state.player_level][block_position[:, 0], block_position[:, 1]]
        == BlockType.STONE.value
    )
    is_mining_stone = jnp.logical_and(
        jnp.logical_and(
            is_block_stone,
            can_mine_stone
        ),
        doing_action,
    )
    is_any_player_mining_stone = jnp.logical_and(
        equal_block_placement,
        is_mining_stone[:, None]
    ).any(axis=0)
    mined_stone_block = jnp.where(
        is_any_player_mining_stone,
        BlockType.PATH.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(mined_stone_block)
    new_inventory = new_inventory.replace(
        stone=new_inventory.stone + 1 * is_mining_stone
    )

    # Furnace
    is_block_furnace = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.FURNACE.value
    )
    is_mining_furnace = jnp.logical_and(
        is_block_furnace,
        doing_action,
    )
    is_any_player_mining_furnace = jnp.logical_and(
        equal_block_placement,
        is_mining_furnace[:, None]
    ).any(axis=0)
    mined_furnace_block = jnp.where(
        is_any_player_mining_furnace,
        BlockType.PATH.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(mined_furnace_block)

    # Crafting Bench
    is_block_crafting_table = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.CRAFTING_TABLE.value
    )
    is_mining_crafting_table = jnp.logical_and(
        is_block_crafting_table,
        doing_action,
    )
    is_any_player_mining_crafting_table = jnp.logical_and(
        equal_block_placement,
        is_mining_crafting_table[:, None]
    ).any(axis=0)
    mined_crafting_table_block = jnp.where(
        is_any_player_mining_crafting_table,
        BlockType.PATH.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(
        mined_crafting_table_block
    )

    # Coal
    can_mine_coal = state.inventory.pickaxe >= 1
    is_block_coal = (
        state.map[state.player_level][block_position[:, 0], block_position[:, 1]]
        == BlockType.COAL.value
    )
    is_mining_coal = jnp.logical_and(
        jnp.logical_and(
            is_block_coal,
            can_mine_coal
        ),
        doing_action,
    )
    is_any_player_mining_coal = jnp.logical_and(
        equal_block_placement,
        is_mining_coal[:, None]
    ).any(axis=0)
    mined_coal_block = jnp.where(
        is_any_player_mining_coal,
        BlockType.PATH.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(mined_coal_block)
    new_inventory = new_inventory.replace(
        coal=new_inventory.coal + 1 * is_mining_coal
    )

    # Iron
    can_mine_iron = state.inventory.pickaxe >= 2
    is_block_iron = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.IRON.value
    )
    is_mining_iron = jnp.logical_and(
        jnp.logical_and(
            is_block_iron,
            can_mine_iron
        ),
        doing_action,
    )
    is_any_player_mining_iron = jnp.logical_and(
        equal_block_placement,
        is_mining_iron[:, None]
    ).any(axis=0)
    mined_iron_block = jnp.where(
        is_any_player_mining_iron,
        BlockType.PATH.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(mined_iron_block)
    new_inventory = new_inventory.replace(
        iron=new_inventory.iron + 1 * is_mining_iron
    )

    # Diamond  
    can_mine_diamond = state.inventory.pickaxe >= 3
    is_block_diamond = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.DIAMOND.value
    )
    is_mining_diamond = jnp.logical_and(
        jnp.logical_and(
            is_block_diamond,
            can_mine_diamond
        ),
        doing_action,
    )
    is_any_player_mining_diamond = jnp.logical_and(
        equal_block_placement,
        is_mining_diamond[:, None]
    ).any(axis=0)
    mined_diamond_block = jnp.where(
        is_any_player_mining_diamond,
        BlockType.PATH.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(mined_diamond_block)
    new_inventory = new_inventory.replace(
        diamond=new_inventory.diamond + 1 * is_mining_diamond
    )

    # Sapphire
    can_mine_sapphire = state.inventory.pickaxe >= 4
    is_block_sapphire = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.SAPPHIRE.value
    )
    is_mining_sapphire = jnp.logical_and(
        jnp.logical_and(
            is_block_sapphire,
            can_mine_sapphire
        ),
        doing_action,
    )
    is_any_player_mining_sapphire = jnp.logical_and(
        equal_block_placement,
        is_mining_sapphire[:, None]
    ).any(axis=0)
    mined_sapphire_block = jnp.where(
        is_any_player_mining_sapphire,
        BlockType.PATH.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(mined_sapphire_block)
    new_inventory = new_inventory.replace(
        sapphire=new_inventory.sapphire + 1 * is_mining_sapphire
    )

    # Ruby
    can_mine_ruby = state.inventory.pickaxe >= 4
    is_block_ruby = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.RUBY.value
    )
    is_mining_ruby = jnp.logical_and(
        jnp.logical_and(
            is_block_ruby,
            can_mine_ruby
        ),
        doing_action,
    )
    is_any_player_mining_ruby = jnp.logical_and(
        equal_block_placement,
        is_mining_ruby[:, None]
    ).any(axis=0)
    mined_ruby_block = jnp.where(
        is_any_player_mining_ruby,
        BlockType.PATH.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(mined_ruby_block)
    new_inventory = new_inventory.replace(
        ruby=new_inventory.ruby + 1 * is_mining_ruby
    )

    # Sapling
    rng, _rng = jax.random.split(rng)
    is_block_grass = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.GRASS.value
    )
    sapling_prob = 0.2 * is_forager # only foragers can collect saplings
    is_mining_sapling = jnp.logical_and(
        jnp.logical_and(
            is_block_grass,
            jax.random.uniform(_rng, (static_params.player_count,)) < sapling_prob,
        ),
        doing_action,
    )

    new_inventory = new_inventory.replace(
        sapling=new_inventory.sapling + 1 * is_mining_sapling
    )

    # Water
    is_block_water = jnp.logical_or(
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.WATER.value,
        state.map[state.player_level][block_position[:, 0], block_position[:, 1]]
        == BlockType.FOUNTAIN.value,
    )
    is_drinking_water = jnp.logical_and(
        is_block_water,
        doing_action,
    )
    is_drinking_water = jnp.logical_and(
        is_drinking_water,
        is_forager
    )
    new_drink = jnp.where(
        is_drinking_water,
        jnp.minimum(get_max_drink(state), state.player_drink + 3),
        state.player_drink,
    )
    new_thirst = jnp.where(
        is_drinking_water, 
        0.0, 
        state.player_thirst
    )
    new_achievements = state.achievements.at[:, Achievement.COLLECT_DRINK.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.COLLECT_DRINK.value], is_drinking_water
        )
    )

    # Plant
    is_block_plant = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.RIPE_PLANT.value
    )
    is_eating_plant = jnp.logical_and(
        is_block_plant,
        doing_action
    )
    is_any_player_eating_plant = jnp.logical_and(
        equal_block_placement,
        is_eating_plant[:, None]
    ).any(axis=0)
    new_plant = jnp.where(
        is_any_player_eating_plant,
        BlockType.PLANT.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(new_plant)
    new_food = jnp.where(
        is_eating_plant,
        jnp.minimum(get_max_food(state), state.player_food + 4),
        state.player_food,
    )
    new_hunger = jnp.where(is_eating_plant, 0.0, state.player_hunger)
    new_achievements = new_achievements.at[:, Achievement.EAT_PLANT.value].set(
        jnp.logical_or(new_achievements[:, Achievement.EAT_PLANT.value], is_eating_plant)
    )
    
    new_growing_plants_age = update_plants_with_eat(
        state, block_position, is_any_player_eating_plant
    )

    # Stalagmite
    can_mine_stalagmite = state.inventory.pickaxe >= 1
    is_block_stalagmite = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.STALAGMITE.value
    )
    is_mining_stalagmite = jnp.logical_and(
        jnp.logical_and(
            is_block_stalagmite,
            can_mine_stalagmite
        ),
        doing_action,
    )
    is_any_player_mining_stalagmite = jnp.logical_and(
        equal_block_placement,
        is_mining_stalagmite[:, None]
    ).any(axis=0)
    mined_stalagmite_block = jnp.where(
        is_any_player_mining_stalagmite,
        BlockType.PATH.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(mined_stalagmite_block)
    new_inventory = new_inventory.replace(
        stone=new_inventory.stone + 1 * is_mining_stalagmite
    )

    # Chest
    is_block_chest = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.CHEST.value
    )
    is_players_chest = (
        (state.chest_positions[state.player_level] == block_position[:, None]).all(axis=-1)
    ).any(axis=-1)
    is_opening_chest = jnp.logical_and(
        is_players_chest,
        jnp.logical_and(
            is_block_chest,
            doing_action,
        )
    )
    is_any_player_opening_chest = jnp.logical_and(
        equal_block_placement,
        is_opening_chest[:, None]
    ).any(axis=0)

    mined_chest_block = jnp.where(
        is_any_player_opening_chest,
        BlockType.PATH.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(mined_chest_block)
    rng, _rng = jax.random.split(rng)
    new_inventory = add_items_from_chest(_rng, state, new_inventory, is_opening_chest)

    new_chests_opened = state.chests_opened.at[state.player_level].set(
        jnp.logical_or(state.chests_opened[state.player_level], is_opening_chest)
    )

    new_achievements = new_achievements.at[:, Achievement.OPEN_CHEST.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.OPEN_CHEST.value], is_opening_chest
        )
    )

    # Boss
    is_attacking_boss = jnp.logical_and(
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.NECROMANCER.value,
        doing_action,
    )
    can_damage_boss = jnp.logical_and(
        is_boss_vulnerable(state), is_fighting_boss(state, static_params)
    )
    is_damaging_boss = jnp.logical_and(
        is_attacking_boss,
        can_damage_boss,
    )

    new_boss_progress = state.boss_progress + 1 * is_damaging_boss.any()
    new_boss_timesteps_to_spawn_this_round = (
        BOSS_FIGHT_SPAWN_TURNS * is_damaging_boss.any()
        + state.boss_timesteps_to_spawn_this_round * (1 - is_damaging_boss.any())
    )

    new_achievements = new_achievements.at[:, Achievement.DAMAGE_NECROMANCER.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.DAMAGE_NECROMANCER.value], is_damaging_boss
        )
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

    return state


def do_crafting(state, actions, static_params):
    is_at_crafting_table = is_near_block(state, BlockType.CRAFTING_TABLE.value, static_params)
    is_at_furnace = is_near_block(state, BlockType.FURNACE.value, static_params)
    is_miner = state.player_specialization == Specialization.MINER.value
    is_warrior = state.player_specialization == Specialization.WARRIOR.value

    new_achievements = state.achievements

    # Wood pickaxe
    can_craft_wood_pickaxe = jnp.logical_and(
        state.inventory.wood >= 1,
        is_miner
    )

    is_crafting_wood_pickaxe = jnp.logical_and(
        actions == Action.MAKE_WOOD_PICKAXE.value,
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
        is_miner,
        jnp.logical_and(
            new_inventory.wood >= 1, new_inventory.stone >= 1
        )
    )
    is_crafting_stone_pickaxe = jnp.logical_and(
        actions == Action.MAKE_STONE_PICKAXE.value,
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
    can_craft_iron_pickaxe = jnp.logical_and(
        is_miner,
        can_craft_iron_pickaxe,
    )
    is_crafting_iron_pickaxe = jnp.logical_and(
        actions == Action.MAKE_IRON_PICKAXE.value,
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
    can_craft_diamond_pickaxe = jnp.logical_and(
        is_miner,
        can_craft_diamond_pickaxe,
    )
    is_crafting_diamond_pickaxe = jnp.logical_and(
        actions == Action.MAKE_DIAMOND_PICKAXE.value,
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
        actions == Action.MAKE_WOOD_SWORD.value,
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
        actions == Action.MAKE_STONE_SWORD.value,
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
    can_craft_iron_sword = jnp.logical_and(
        is_warrior,
        can_craft_iron_sword,
    )
    is_crafting_iron_sword = jnp.logical_and(
        actions == Action.MAKE_IRON_SWORD.value,
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
    can_craft_diamond_sword = jnp.logical_and(
        is_warrior,
        can_craft_diamond_sword,
    )
    is_crafting_diamond_sword = jnp.logical_and(
        actions == Action.MAKE_DIAMOND_SWORD.value,
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
    can_craft_iron_armour = (new_inventory.armour < 1).sum(axis=1) > 0
    can_craft_iron_armour = jnp.logical_and(
        can_craft_iron_armour,
        jnp.logical_and(new_inventory.iron >= 3, new_inventory.coal >= 3),
    )

    iron_armour_index_to_craft = jnp.argmax(new_inventory.armour < 1, axis=1)

    is_crafting_iron_armour = jnp.logical_and(
        actions == Action.MAKE_IRON_ARMOUR.value,
        jnp.logical_and(
            can_craft_iron_armour,
            jnp.logical_and(is_at_crafting_table, is_at_furnace),
        ),
    )

    new_inventory = new_inventory.replace(
        iron=new_inventory.iron - 3 * is_crafting_iron_armour,
        coal=new_inventory.coal - 3 * is_crafting_iron_armour,
        armour=new_inventory.armour.at[
            jnp.arange(0, len(new_inventory.armour)),
            iron_armour_index_to_craft
        ].set(
            is_crafting_iron_armour * 1
            + (1 - is_crafting_iron_armour)
            * new_inventory.armour[
                jnp.arange(0, len(new_inventory.armour)),
                iron_armour_index_to_craft
            ]
        ),
    )
    new_achievements = new_achievements.at[:, Achievement.MAKE_IRON_ARMOUR.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.MAKE_IRON_ARMOUR.value],
            is_crafting_iron_armour,
        )
    )

    # Diamond armour
    can_craft_diamond_armour = (new_inventory.armour < 2).sum(axis=1) > 0
    can_craft_diamond_armour = jnp.logical_and(
        can_craft_diamond_armour, new_inventory.diamond >= 3
    )

    diamond_armour_index_to_craft = jnp.argmax(new_inventory.armour < 2, axis=1)

    is_crafting_diamond_armour = jnp.logical_and(
        actions == Action.MAKE_DIAMOND_ARMOUR.value,
        jnp.logical_and(
            can_craft_diamond_armour,
            is_at_crafting_table,
        ),
    )

    new_inventory = new_inventory.replace(
        diamond=new_inventory.diamond - 3 * is_crafting_diamond_armour,
        armour=new_inventory.armour.at[
            jnp.arange(0, len(new_inventory.armour)),
            diamond_armour_index_to_craft
        ].set(
            is_crafting_diamond_armour * 2
            + (1 - is_crafting_diamond_armour)
            * new_inventory.armour[
                jnp.arange(0, len(new_inventory.armour)),
                diamond_armour_index_to_craft
            ]
        ),
    )
    new_achievements = new_achievements.at[:, Achievement.MAKE_DIAMOND_ARMOUR.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.MAKE_DIAMOND_ARMOUR.value],
            is_crafting_diamond_armour,
        )
    )

    # Arrow
    can_craft_arrow = jnp.logical_and(new_inventory.stone >= 1, new_inventory.wood >= 1)
    can_craft_arrow = jnp.logical_and(
        can_craft_arrow,
        is_warrior
    )
    is_crafting_arrow = jnp.logical_and(
        actions == Action.MAKE_ARROW.value,
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
    can_craft_torch = jnp.logical_and(
        can_craft_torch,
        is_miner,
    )
    is_crafting_torch = jnp.logical_and(
        actions == Action.MAKE_TORCH.value,
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


def add_new_growing_plant(growing_plant_positions, growing_plant_age, growing_plant_mask, position, is_placing_sapling):
    is_empty = jnp.logical_not(growing_plant_mask)
    plant_index = jnp.argmax(is_empty)
    is_an_empty_slot = is_empty.any()
    is_adding_plant = jnp.logical_and(is_an_empty_slot, is_placing_sapling)

    new_growing_plants_positions = jax.lax.select(
        is_adding_plant,
        growing_plant_positions.at[plant_index].set(position),
        growing_plant_positions,
    )
    new_growing_plants_age = jax.lax.select(
        is_adding_plant,
        growing_plant_age.at[plant_index].set(0),
        growing_plant_age,
    )
    new_growing_plants_mask = jax.lax.select(
        is_adding_plant,
        growing_plant_mask.at[plant_index].set(True),
        growing_plant_mask,
    )
    return new_growing_plants_positions, new_growing_plants_age, new_growing_plants_mask, is_adding_plant


def place_block(state, action, static_params):
    placing_block_position = state.player_position + DIRECTIONS[state.player_direction]
    equal_block_placement = (jnp.expand_dims(placing_block_position, axis=1) == jnp.expand_dims(placing_block_position, axis=0)).all(axis=2)

    new_map = state.map[state.player_level]
    new_item_map = state.item_map[state.player_level]

    is_block_in_other_player = is_in_other_player(state, placing_block_position)
    is_block_in_mob = is_in_mob(state, placing_block_position)
    is_block_in_bounds = in_bounds(placing_block_position, static_params)
    is_placement_in_bounds_not_in_mobs = jnp.logical_and(
        is_block_in_bounds,
        jnp.logical_not(jnp.logical_or(
            is_block_in_other_player,
            is_block_in_mob
        ))
    )

    # Crafting table
    is_valid_placement = jnp.logical_and(
        is_placement_in_bounds_not_in_mobs,
        jnp.logical_and(
            jnp.logical_not(is_in_solid_block(new_map, placing_block_position)),
            new_item_map[placing_block_position[:, 0], placing_block_position[:, 1]]
            == ItemType.NONE.value,
        )
    )
    crafting_table_key_down = action == Action.PLACE_TABLE.value
    has_wood = state.inventory.wood >= 2
    is_player_placing_crafting_table = jnp.logical_and(
        crafting_table_key_down,
        jnp.logical_and(is_valid_placement, has_wood),
    )
    is_any_player_placing_crafting_table = jnp.logical_and(
        equal_block_placement,
        is_player_placing_crafting_table[:, None]
    ).any(axis=0)

    placed_crafting_table_block = jnp.where(
        is_any_player_placing_crafting_table,
        BlockType.CRAFTING_TABLE.value,
        new_map[
            placing_block_position[:, 0], placing_block_position[:, 1]
        ],
    )
    new_map = (
        new_map
        .at[placing_block_position[:, 0], placing_block_position[:, 1]]
        .set(placed_crafting_table_block)
    )
    new_inventory = state.inventory.replace(
        wood=state.inventory.wood - 2 * is_player_placing_crafting_table
    )
    new_achievements = state.achievements.at[:, Achievement.PLACE_TABLE.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.PLACE_TABLE.value], is_player_placing_crafting_table
        )
    )


    # Furnace
    is_valid_placement = jnp.logical_and(
        is_placement_in_bounds_not_in_mobs,
        jnp.logical_and(
            jnp.logical_not(is_in_solid_block(new_map, placing_block_position)),
            new_item_map[placing_block_position[:, 0], placing_block_position[:, 1]]
            == ItemType.NONE.value,
        )
    )

    furnace_key_down = action == Action.PLACE_FURNACE.value
    has_stone = new_inventory.stone > 0
    is_player_placing_furnace = jnp.logical_and(
        furnace_key_down,
        jnp.logical_and(
            is_valid_placement, has_stone
        ),
    )
    is_any_player_placing_furnace = jnp.logical_and(
        equal_block_placement,
        is_player_placing_furnace[:, None]
    ).any(axis=0)
    placed_furnace_block = jnp.where(
        is_any_player_placing_furnace,
        BlockType.FURNACE.value,
        new_map[placing_block_position[:, 0], placing_block_position[:, 1]],
    )
    new_map = new_map.at[placing_block_position[:, 0], placing_block_position[:, 1]].set(
        placed_furnace_block
    )
    new_inventory = new_inventory.replace(
        stone=new_inventory.stone - 1 * is_player_placing_furnace
    )
    new_achievements = new_achievements.at[:, Achievement.PLACE_FURNACE.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.PLACE_FURNACE.value], is_player_placing_furnace
        )
    )

    # Stone
    stone_key_down = action == Action.PLACE_STONE.value
    has_stone = new_inventory.stone > 0
    is_valid_placement = jnp.logical_and(
        is_placement_in_bounds_not_in_mobs,
        new_item_map[placing_block_position[:, 0], placing_block_position[:, 1]]
        == ItemType.NONE.value,
    )
    is_valid_placement = jnp.logical_and(
        is_valid_placement,
        jnp.logical_or(
            new_map[
                placing_block_position[:, 0], placing_block_position[:, 1]
            ]
            == BlockType.WATER.value,
            jnp.logical_not(is_in_solid_block(new_map, placing_block_position)),
        )
    )
    is_player_placing_stone = jnp.logical_and(
        stone_key_down,
        jnp.logical_and(is_valid_placement, has_stone),
    )
    is_player_placing_stone = jnp.logical_and(
        is_player_placing_stone,
        state.player_specialization == Specialization.MINER.value
    )
    is_any_player_placing_stone = jnp.logical_and(
        equal_block_placement,
        is_player_placing_stone[:, None]
    ).any(axis=0)
    placed_stone_block = jnp.where(
        is_any_player_placing_stone,
        BlockType.STONE.value,
        new_map[placing_block_position[:, 0], placing_block_position[:, 1]],
    )
    new_map = new_map.at[placing_block_position[:, 0], placing_block_position[:, 1]].set(
        placed_stone_block
    )
    new_inventory = new_inventory.replace(
        stone=new_inventory.stone - 1 * is_player_placing_stone
    )
    new_achievements = new_achievements.at[:, Achievement.PLACE_STONE.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.PLACE_STONE.value], is_player_placing_stone
        )
    )

    # Torch
    # TODO: Make more parallelized
    def _player_place_torch(action_info, player_index):
        (
            working_item_map,
            working_padded_light_map,
        ) = action_info

        torch_key_down = action[player_index] == Action.PLACE_TORCH.value
        has_torch = new_inventory.torches[player_index] > 0
        
        is_valid_placement = jnp.logical_and(
            CAN_PLACE_ITEM_MAPPING[
                new_map[
                    placing_block_position[player_index, 0], placing_block_position[player_index, 1]
                ]
            ],
            working_item_map[placing_block_position[player_index, 0], placing_block_position[player_index, 1]]
            == ItemType.NONE.value,
        )
        is_valid_placement = jnp.logical_and(
            is_placement_in_bounds_not_in_mobs[player_index],
            is_valid_placement
        )
        is_player_placing_torch = jnp.logical_and(
            torch_key_down,
            jnp.logical_and(is_valid_placement, has_torch),
        )
        placed_torch_item = jax.lax.select(
            is_player_placing_torch,
            ItemType.TORCH.value,
            working_item_map[placing_block_position[player_index, 0], placing_block_position[player_index, 1]],
        )
        working_item_map = working_item_map.at[
            placing_block_position[player_index, 0], placing_block_position[player_index, 1]
        ].set(placed_torch_item)

        current_light_map = jax.lax.dynamic_slice(
            working_padded_light_map,
            placing_block_position[player_index]
            - jnp.array([4, 4])
            + jnp.array([light_map_padding, light_map_padding]),
            (9, 9),
        )
        torch_light_map = jnp.clip(TORCH_LIGHT_MAP + current_light_map, 0.0, 1.0)
        torch_light_map = torch_light_map * is_player_placing_torch + current_light_map * (
            1 - is_player_placing_torch
        )
        working_padded_light_map = jax.lax.dynamic_update_slice(
            working_padded_light_map,
            torch_light_map,
            placing_block_position[player_index]
            - jnp.array([4, 4])
            + jnp.array([light_map_padding, light_map_padding]),
        )
        return (working_item_map, working_padded_light_map), is_player_placing_torch
    
    light_map_padding = 6
    padded_light_map_floor = jnp.pad(
        state.light_map[state.player_level],
        (light_map_padding, light_map_padding),
        constant_values=0,
    )
    (new_item_map, padded_light_map_floor), is_player_placing_torch = jax.lax.scan(
        _player_place_torch,
        (new_item_map, padded_light_map_floor),
        jnp.arange(static_params.player_count)
    )
    new_light_map_floor = padded_light_map_floor[
        light_map_padding:-light_map_padding, light_map_padding:-light_map_padding
    ]
    new_light_map = state.light_map.at[state.player_level].set(new_light_map_floor)

    new_inventory = new_inventory.replace(
        torches=new_inventory.torches - 1 * is_player_placing_torch
    )
    new_achievements = new_achievements.at[:, Achievement.PLACE_TORCH.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.PLACE_TORCH.value], is_player_placing_torch
        )
    )


    # Plant
    # TODO: Make more parallelized
    def _player_place_plant(action_info, player_index):
        (
            working_map, 
            working_growing_plants_positions,
            working_growing_plants_age,
            working_growing_plants_mask,
        ) = action_info
        sapling_key_down = action[player_index] == Action.PLACE_PLANT.value
        has_sapling = state.inventory.sapling[player_index] > 0
        is_valid_placement = jnp.logical_and(
            is_placement_in_bounds_not_in_mobs[player_index],
            jnp.logical_and(
                working_map[placing_block_position[player_index, 0], placing_block_position[player_index, 1]]
                == BlockType.GRASS.value,
                new_item_map[placing_block_position[player_index, 0], placing_block_position[player_index, 1]]
                == ItemType.NONE.value,
            )
        )
        is_player_placing_sapling = jnp.logical_and(
            is_valid_placement,
            jnp.logical_and(
                sapling_key_down,
                has_sapling,
            )
        )
        (
            working_growing_plants_positions,
            working_growing_plants_age,
            working_growing_plants_mask,
            is_player_placing_sapling
        ) = add_new_growing_plant(
            working_growing_plants_positions,
            working_growing_plants_age,
            working_growing_plants_mask,
            placing_block_position[player_index], 
            is_player_placing_sapling,
        )
        placed_sapling_block = jax.lax.select(
            is_player_placing_sapling,
            BlockType.PLANT.value,
            working_map[placing_block_position[player_index, 0], placing_block_position[player_index, 1]],
        )
        working_map = working_map.at[placing_block_position[player_index, 0], placing_block_position[player_index, 1]].set(
            placed_sapling_block
        )
        return (
            working_map,
            working_growing_plants_positions,
            working_growing_plants_age,
            working_growing_plants_mask,
        ), is_player_placing_sapling

    (new_map, new_growing_plants_positions, new_growing_plants_age, new_growing_plants_mask), is_player_placing_sapling = jax.lax.scan(
        _player_place_plant,
        (new_map, state.growing_plants_positions, state.growing_plants_age, state.growing_plants_mask),
        jnp.arange(static_params.player_count)
    )

    new_inventory = new_inventory.replace(
        sapling=new_inventory.sapling - 1 * is_player_placing_sapling
    )
    new_achievements = new_achievements.at[:, Achievement.PLACE_PLANT.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.PLACE_PLANT.value], is_player_placing_sapling
        )
    )

    # Do?
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


def update_mobs(rng, state, params, env_params, static_params):

    # Move melee_mobs
    def _move_melee_mob(rng_and_state, melee_mob_index):
        rng, state = rng_and_state
        melee_mobs = state.melee_mobs

        # Random move
        rng, _rng = jax.random.split(rng)
        valid_random_moves = in_bounds(
            DIRECTIONS[1:5] + melee_mobs.position[state.player_level, melee_mob_index], 
            static_params
        )
        random_move_direction = jax.random.choice(
            _rng,
            DIRECTIONS[1:5],
            p=valid_random_moves
        )
        random_move_proposed_position = (
            melee_mobs.position[state.player_level, melee_mob_index]
            + random_move_direction
        )

        # Move towards closest player
        player_move_direction = jnp.zeros((2,), dtype=jnp.int32)
        all_players_move_direction_abs = jnp.abs(
            state.player_position
            - melee_mobs.position[state.player_level, melee_mob_index]
        )
        distance_to_players = all_players_move_direction_abs.sum(axis=1)
        player_targetted = jnp.argmin(jnp.where(
            state.player_alive,
            distance_to_players,
            jnp.inf
        ))
        player_move_direction_abs = all_players_move_direction_abs[player_targetted]

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
                state.player_position[player_targetted, player_move_direction_index]
                - melee_mobs.position[state.player_level, melee_mob_index, player_move_direction_index]
            ).astype(jnp.int32)
        )
        player_move_proposed_position = (
            melee_mobs.position[state.player_level, melee_mob_index]
            + player_move_direction
        )

        # Choose movement
        close_to_player = distance_to_players < 10
        close_to_player = jnp.logical_and(
            close_to_player,
            state.player_alive
        ).any()
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
        is_attacking_player = distance_to_players == 1
        is_attacking_player = jnp.logical_and(
            is_attacking_player,
            state.player_alive
        )
        is_attacking_player = jnp.logical_and(
            is_attacking_player,
            melee_mobs.attack_cooldown[state.player_level, melee_mob_index] <= 0,
        )
        is_attacking_player = jnp.logical_and(
            is_attacking_player, melee_mobs.mask[state.player_level, melee_mob_index]
        )

        proposed_position = jax.lax.select(
            is_attacking_player.any(),
            melee_mobs.position[state.player_level, melee_mob_index],
            proposed_position,
        )

        melee_mob_base_damage = MOB_TYPE_DAMAGE_MAPPING[
            melee_mobs.type_id[state.player_level, melee_mob_index], MobType.MELEE.value
        ]

        melee_mob_damage = get_damage_done_to_player(
            state, static_params, melee_mob_base_damage * (1 + 2.5 * state.is_sleeping[:, None])
        )

        new_cooldown = jnp.where(
            is_attacking_player.any(),
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
            achievements=state.achievements.at[:, Achievement.WAKE_UP.value].set(
                jnp.logical_or(
                    state.achievements[:, Achievement.WAKE_UP.value], is_waking_player
                )
            ),
        )

        mob_type = melee_mobs.type_id[state.player_level, melee_mob_index]
        collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type, 1]
        valid_move = is_position_in_bounds_not_in_mob_not_colliding(
            state, proposed_position[None, :], collision_map, static_params
        )[0]
        in_other_player = is_in_other_player(state, proposed_position[None, :])[0]
        valid_move = jnp.logical_and(
            valid_move,
            jnp.logical_not(in_other_player)
        )

        
        position = jax.lax.select(
            valid_move,
            proposed_position,
            melee_mobs.position[state.player_level, melee_mob_index],
        )

        should_not_despawn = distance_to_players < params.mob_despawn_distance
        should_not_despawn = jnp.logical_and(
            should_not_despawn,
            state.player_alive
        ).any()
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
        _move_melee_mob, (rng, state), jnp.arange(static_params.max_melee_mobs * static_params.player_count)
    )

    # Move passive_mobs
    def _move_passive_mob(rng_and_state, passive_mob_index):
        rng, state = rng_and_state
        passive_mobs = state.passive_mobs

        # Random move
        rng, _rng = jax.random.split(rng)
        valid_random_moves = in_bounds(
            DIRECTIONS[1:9] + passive_mobs.position[state.player_level, passive_mob_index], 
            static_params
        )
        random_move_direction = jax.random.choice(
            _rng,
            DIRECTIONS[1:9],  # 50% chance of not moving
            p=valid_random_moves
        )
        proposed_position = (
            passive_mobs.position[state.player_level, passive_mob_index]
            + random_move_direction
        )

        mob_type = passive_mobs.type_id[state.player_level, passive_mob_index]
        collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type, 0]
        valid_move = is_position_in_bounds_not_in_mob_not_colliding(
            state, proposed_position[None, :], collision_map, static_params
        )[0]
        in_other_player = is_in_other_player(state, proposed_position[None, :])[0]
        valid_move = jnp.logical_and(
            valid_move,
            jnp.logical_not(in_other_player)
        )
        position = jax.lax.select(
            valid_move,
            proposed_position,
            passive_mobs.position[state.player_level, passive_mob_index],
        )

        distance_to_players = jnp.abs(
            state.player_position
            - passive_mobs.position[state.player_level, passive_mob_index]
        ).sum(axis=1)
        should_not_despawn = jnp.logical_and(
            distance_to_players < params.mob_despawn_distance,
            state.player_alive
        ).any()

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
        _move_passive_mob, (rng, state), jnp.arange(static_params.max_passive_mobs * static_params.player_count)
    )

    # Move ranged_mobs

    def _move_ranged_mob(rng_and_state, ranged_mob_index):
        rng, state = rng_and_state
        ranged_mobs = state.ranged_mobs

        # Random move
        rng, _rng = jax.random.split(rng)
        valid_random_moves = in_bounds(
            DIRECTIONS[1:5] + ranged_mobs.position[state.player_level, ranged_mob_index], 
            static_params
        )
        random_move_direction = jax.random.choice(
            _rng,
            DIRECTIONS[1:5],
            p=valid_random_moves
        )
        random_move_proposed_position = (
            ranged_mobs.position[state.player_level, ranged_mob_index]
            + random_move_direction
        )

        # Move towards closest player
        player_move_direction = jnp.zeros((2,), dtype=jnp.int32)
        all_players_move_direction_abs = jnp.abs(
            state.player_position
            - ranged_mobs.position[state.player_level, ranged_mob_index]
        )
        distance_to_players = all_players_move_direction_abs.sum(axis=1)
        player_targetted = jnp.argmin(jnp.where(
            state.player_alive,
            distance_to_players,
            jnp.inf
        ))
        player_move_direction_abs = all_players_move_direction_abs[player_targetted]
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
                state.player_position[player_targetted, player_move_direction_index]
                - ranged_mobs.position[state.player_level, ranged_mob_index, player_move_direction_index]
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
        far_from_player = player_move_direction_abs[player_move_direction_index] >= 6
        too_close_to_player = player_move_direction_abs[player_move_direction_index] <= 3

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
            jax.random.uniform(_rng) < 0.15,
            proposed_position,
            random_move_proposed_position,
        )

        # Choose attack or not
        is_attacking_player = jnp.logical_not(far_from_player)
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
            < static_params.max_mob_projectiles * static_params.player_count
        )
        new_projectile_position = ranged_mobs.position[
            state.player_level, ranged_mob_index
        ]

        is_spawning_projectile = jnp.logical_and(
            is_attacking_player, can_spawn_projectile
        )

        new_mob_projectiles, new_mob_projectile_directions, new_mob_projectile_owners = spawn_projectile(
            state,
            static_params,
            state.mob_projectiles,
            state.mob_projectile_directions,
            state.mob_projectile_owners,
            new_projectile_position,
            is_spawning_projectile,
            ranged_mob_index,
            player_move_direction,
            RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_MAPPING[
                ranged_mobs.type_id[state.player_level, ranged_mob_index]
            ],
        )

        state = state.replace(
            mob_projectiles=new_mob_projectiles,
            mob_projectile_directions=new_mob_projectile_directions,
            mob_projectile_owners=new_mob_projectile_owners,
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

        mob_type = ranged_mobs.type_id[state.player_level, ranged_mob_index]
        collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type, 2]
        valid_move = is_position_in_bounds_not_in_mob_not_colliding(
            state, proposed_position[None, :], collision_map, static_params
        )[0]
        in_other_player = is_in_other_player(state, proposed_position[None, :])[0]
        valid_move = jnp.logical_and(
            valid_move,
            jnp.logical_not(in_other_player)
        )

        position = jax.lax.select(
            valid_move,
            proposed_position,
            ranged_mobs.position[state.player_level, ranged_mob_index],
        )

        should_not_despawn = distance_to_players < params.mob_despawn_distance
        should_not_despawn = jnp.logical_and(
            should_not_despawn,
            state.player_alive
        ).any()
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
        _move_ranged_mob, (rng, state), jnp.arange(static_params.max_ranged_mobs * static_params.player_count)
    )

    # Move projectiles
    def _move_mob_projectile(rng_and_state, projectile_index):
        rng, state = rng_and_state
        projectiles = state.mob_projectiles

        proposed_position = (
            projectiles.position[state.player_level, projectile_index]
            + state.mob_projectile_directions[state.player_level, projectile_index]
        )


        proposed_position_in_bounds = in_bounds(proposed_position[None, :], static_params)[0]
        in_wall = is_in_solid_block(state.map[state.player_level], proposed_position[None, :])[0]
        in_wall = jnp.logical_and(
            in_wall,
            jnp.logical_not(
                state.map[state.player_level][
                    proposed_position[0], proposed_position[1]
                ]
                == BlockType.WATER.value
            ),
        )  # Arrows can go over water
        in_mob = is_in_mob(state, proposed_position[None, :])[0]

        continue_move = jnp.logical_and(
            proposed_position_in_bounds, jnp.logical_not(in_wall)
        )
        continue_move = jnp.logical_and(continue_move, jnp.logical_not(in_mob))

        hit_player0 = jnp.logical_and(
            (
                projectiles.position[state.player_level, projectile_index]
                == state.player_position
            ).all(axis=1),
            projectiles.mask[state.player_level, projectile_index],
        )

        proposed_position_in_player = (proposed_position == state.player_position).all(axis=1)
        hit_player1 = jnp.logical_and(
            proposed_position_in_player,
            projectiles.mask[state.player_level, projectile_index],
        )
        hit_player = jnp.logical_or(hit_player0, hit_player1)
        hit_player = jnp.logical_and(hit_player, state.player_alive)

        continue_move = jnp.logical_and(continue_move, jnp.logical_not(hit_player.any()))

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
            MOB_TYPE_DAMAGE_MAPPING[projectile_type, MobType.PROJECTILE.value][None, :],
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
        jnp.arange(static_params.max_mob_projectiles * static_params.player_count),
    )

    def _move_player_projectile(rng_and_state, projectile_index):
        rng, state = rng_and_state
        projectiles = state.player_projectiles

        projectile_owner = state.player_projectile_owners[
            state.player_level, projectile_index
        ]

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
        arrow_damage_add = arrow_damage_add.at[state.bow_enchantment[projectile_owner]].set(
            projectile_damage_vector[0] / 2
        )
        arrow_damage_add = arrow_damage_add.at[0].set(0)

        projectile_damage_vector += jax.lax.select(
            is_arrow,
            arrow_damage_add,
            jnp.zeros(3, dtype=jnp.float32),
        )

        # Apply attribute scaling
        arrow_damage_coeff = 1 + 0.2 * (state.player_dexterity[projectile_owner] - 1)
        magic_damage_coeff = 1 + 0.5 * (state.player_intelligence[projectile_owner] - 1)

        projectile_damage_vector *= jax.lax.select(
            is_arrow,
            arrow_damage_coeff,
            1.0,
        )

        projectile_damage_vector *= jax.lax.select(
            projectile_type == ProjectileType.FIREBALL.value,
            magic_damage_coeff,
            1.0,
        )

        proposed_position = (
            projectiles.position[state.player_level, projectile_index]
            + state.player_projectile_directions[state.player_level, projectile_index]
        )
        
        proposed_position_in_bounds = in_bounds(proposed_position[None, :], static_params)[0]
        in_wall = is_in_solid_block(state.map[state.player_level], proposed_position[None, :])[0]
        in_wall = jnp.logical_and(
            in_wall,
            jnp.logical_not(
                state.map[state.player_level][
                    proposed_position[0], proposed_position[1]
                ]
                == BlockType.WATER.value
            ),
        )  # Arrows can go over water

        # Check if we hit a player
        deal_damage = projectiles.mask[state.player_level, projectile_index]

        per_player_contact = (state.player_position == proposed_position[None, :]).all(axis=-1)
        did_attack_player = per_player_contact.any()    
        player_attack_index = jnp.argmax(per_player_contact)

        player_defense_vector = get_player_defense_vector(state)[player_attack_index]
        player_damage_dealt = get_damage(projectile_damage_vector, player_defense_vector) * did_attack_player * env_params.friendly_fire
        new_player_health = state.player_health.at[player_attack_index].subtract(player_damage_dealt)
        
        state, did_attack_mob0, did_kill_mob0 = attack_mob(
            state,
            deal_damage,
            projectiles.position[None, state.player_level, projectile_index],
            projectile_damage_vector[None, :],
            jnp.array([False]),
        )
        did_attack_mob0 = did_attack_mob0[0]

        did_attack_mob = jnp.logical_or(did_attack_player, did_attack_mob0)

        projectile_damage_vector = projectile_damage_vector * (1 - did_attack_mob0)

        state, did_attack_mob1, did_kill_mob1 = attack_mob(
            state,
            deal_damage,
            proposed_position[None, :],
            projectile_damage_vector[None, :],
            jnp.array([False])
        )
        did_attack_mob1 = did_attack_mob1[0]

        did_attack_mob = jnp.logical_or(did_attack_mob, did_attack_mob1)

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
            player_health=new_player_health,
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
        jnp.arange(static_params.max_player_projectiles * static_params.player_count),
    )

    return state


def update_player_intrinsics(state, action, static_params):
    # Start sleeping?
    is_starting_sleep = jnp.logical_and(
        action == Action.SLEEP.value, state.player_energy < get_max_energy(state)
    )
    new_is_sleeping = jnp.logical_or(state.is_sleeping, is_starting_sleep)
    state = state.replace(
        is_sleeping=jnp.where(state.player_alive, new_is_sleeping, state.is_sleeping)
    )

    # Wake up?
    is_waking_up = jnp.logical_and(
        state.player_energy >= get_max_energy(state), state.is_sleeping
    )
    new_is_sleeping = jnp.logical_and(state.is_sleeping, jnp.logical_not(is_waking_up))
    new_achievements = state.achievements.at[:, Achievement.WAKE_UP.value].set(
        jnp.logical_or(state.achievements[:, Achievement.WAKE_UP.value], is_waking_up)
    )
    state = state.replace(
        is_sleeping=jnp.where(state.player_alive, new_is_sleeping, state.is_sleeping),
        achievements=jnp.where(state.player_alive[:, None], new_achievements, state.achievements),
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
        is_resting=jnp.where(state.player_alive, new_is_resting, state.is_resting),
    )

    not_boss = jnp.logical_not(is_fighting_boss(state, static_params))

    intrinsic_decay_coeff = 1.0 - (0.125 * (state.player_dexterity - 1))

    # Hunger
    hunger_add = jnp.where(
        state.is_sleeping, 
        0.5, 
        1.0,
    ) * intrinsic_decay_coeff
    new_hunger = state.player_hunger + hunger_add

    hungered_food = jnp.maximum(state.player_food - 1 * not_boss, 0)
    new_food = jnp.where(new_hunger > 25, hungered_food, state.player_food)
    new_hunger = jnp.where(
        new_hunger > 25, 
        0.0, 
        new_hunger
    )

    state = state.replace(
        player_hunger=jnp.where(state.player_alive, new_hunger, state.player_hunger),
        player_food=jnp.where(state.player_alive, new_food, state.player_food),
    )

    # Thirst
    thirst_add = jnp.where(
        state.is_sleeping, 
        0.5, 
        1.0,
    ) * intrinsic_decay_coeff
    new_thirst = state.player_thirst + thirst_add
    thirsted_drink = jnp.maximum(state.player_drink - 1 * not_boss, 0)
    new_drink = jnp.where(new_thirst > 20, thirsted_drink, state.player_drink)
    new_thirst = jnp.where(
        new_thirst > 20, 
        0.0, 
        new_thirst
    )

    state = state.replace(
        player_thirst=jnp.where(state.player_alive, new_thirst, state.player_thirst),
        player_drink=jnp.where(state.player_alive, new_drink, state.player_drink),
    )

    # Fatigue
    new_fatigue = jnp.where(
        state.is_sleeping,
        jnp.minimum(state.player_fatigue - 1, 0),
        state.player_fatigue + intrinsic_decay_coeff,
    )

    new_energy = jnp.where(
        new_fatigue > 30,
        jnp.maximum(state.player_energy - 1 * not_boss, 0),
        state.player_energy,
    )
    new_fatigue = jnp.where(
        new_fatigue > 30, 
        0.0, 
        new_fatigue
    )

    new_energy = jnp.where(
        new_fatigue < -10,
        jnp.minimum(state.player_energy + 1, get_max_energy(state)),
        new_energy,
    )
    new_fatigue = jnp.where(
        new_fatigue < -10, 
        0.0, 
        new_fatigue
    )

    state = state.replace(
        player_fatigue=jnp.where(state.player_alive, new_fatigue, state.player_fatigue),
        player_energy=jnp.where(state.player_alive, new_energy, state.player_energy),
    )

    # Health
    necessities = jnp.stack(
        [
            state.player_food > 0,
            state.player_drink > 0,
            jnp.logical_or(state.player_energy > 0, state.is_sleeping)
        ],
        axis=1
    )

    all_necessities = necessities.all(axis=1)
    recover_all = jnp.where(
        state.is_sleeping, 
        2.0, 
        1.0,
    )
    recover_not_all = jnp.where(
        state.is_sleeping, 
        -0.5, 
        -1.0,
    ) * not_boss
    recover_add = jnp.where(all_necessities, recover_all, recover_not_all)

    new_recover = state.player_recover + recover_add

    recovered_health = jnp.minimum(state.player_health + 1, get_max_health(state))
    derecovered_health = state.player_health - 1

    new_health = jnp.where(new_recover > 25, recovered_health, state.player_health)
    new_recover = jnp.where(
        new_recover > 25, 
        0.0, 
        new_recover
    )
    new_health = jnp.where(new_recover < -15, derecovered_health, new_health)
    new_recover = jnp.where(
        new_recover < -15, 
        0.0, 
        new_recover
    )

    state = state.replace(
        player_recover=jnp.where(state.player_alive, new_recover, state.player_recover),
        player_health=jnp.where(state.player_alive, new_health, state.player_health),
    )

    # Mana
    mana_recover_coeff = 1 + 0.25 * (state.player_intelligence - 1)
    new_recover_mana = (
        jnp.where(
            state.is_sleeping,
            state.player_recover_mana + 2,
            state.player_recover_mana + 1,
        )
        * mana_recover_coeff
    )

    new_mana = jnp.where(
        new_recover_mana > 30, state.player_mana + 1, state.player_mana
    )
    new_recover_mana = jnp.where(
        new_recover_mana > 30, 
        0.0, 
        new_recover_mana
    )

    state = state.replace(
        player_recover_mana=jnp.where(state.player_alive, new_recover_mana, state.player_recover_mana),
        player_mana=jnp.where(state.player_alive, new_mana, state.player_mana),
    )

    return state


def update_plants(state, static_params):
    growing_plants_age = state.growing_plants_age + 1
    growing_plants_age *= state.growing_plants_mask

    finished_growing_plants = growing_plants_age >= 500

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
        jnp.arange(static_params.max_growing_plants * static_params.player_count),
    )

    new_whole_map = state.map.at[0].set(new_map)

    state = state.replace(
        map=new_whole_map,
        growing_plants_age=growing_plants_age,
    )

    return state


def move_player(state, actions, params, static_params):
    proposed_position = state.player_position + DIRECTIONS[actions]

    valid_move = is_position_in_bounds_not_in_mob_not_colliding(
        state, proposed_position, COLLISION_LAND_CREATURE, static_params
    ) 
    valid_move = jnp.logical_and(valid_move, is_position_not_colliding_other_player(state, proposed_position))
    valid_move = jnp.logical_or(valid_move, params.god_mode)

    position = state.player_position + jnp.expand_dims(valid_move, axis=1).astype(jnp.int32) * DIRECTIONS[actions]

    is_new_direction = jnp.sum(jnp.abs(DIRECTIONS[actions]), axis=1) != 0
    new_direction = (
        state.player_direction * (1 - is_new_direction) + actions * is_new_direction
    )

    state = state.replace(
        player_position=position,
        player_direction=new_direction,
    )

    return state


def spawn_mobs(state, rng, params, static_params):
    player_distance_map = get_all_players_distance_map(
        state.player_position, state.player_alive, static_params
    )
    grave_map = jnp.logical_or(
        state.map[state.player_level] == BlockType.GRAVE.value,
        jnp.logical_or(
            state.map[state.player_level] == BlockType.GRAVE2.value,
            state.map[state.player_level] == BlockType.GRAVE3.value,
        ),
    )

    floor_mob_spawn_chance = FLOOR_MOB_SPAWN_CHANCE * static_params.player_count
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
        < static_params.max_passive_mobs * static_params.player_count
    )

    rng, _rng = jax.random.split(rng)
    can_spawn_passive_mob = jnp.logical_and(
        can_spawn_passive_mob,
        jax.random.uniform(_rng) < floor_mob_spawn_chance[state.player_level, 0],
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

    # To avoid spawning mobs ontop of dead players
    passive_mobs_can_spawn_map = passive_mobs_can_spawn_map.at[
        state.player_position[:, 0], state.player_position[:, 1]
    ].set(False)

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
        state.melee_mobs.mask[state.player_level].sum() < static_params.max_melee_mobs * static_params.player_count
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
    melee_mob_spawn_chance = floor_mob_spawn_chance[
        state.player_level, 1
    ] + floor_mob_spawn_chance[state.player_level, 3] * jnp.square(
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
    melee_mobs_can_spawn_map = melee_mobs_can_spawn_map.at[
        state.player_position[:, 0], state.player_position[:, 1]
    ].set(False)

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
        state.ranged_mobs.mask[state.player_level].sum() < static_params.max_ranged_mobs * static_params.player_count
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
        < floor_mob_spawn_chance[state.player_level, 2] * monster_spawn_coeff,
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
    ranged_mobs_can_spawn_map = ranged_mobs_can_spawn_map.at[
        state.player_position[:, 0], state.player_position[:, 1]
    ].set(False)

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
    state: EnvState, actions, env_params: EnvParams, static_params: StaticEnvParams
):
    is_moving_down = jnp.logical_and(
        actions == Action.DESCEND.value,
        jnp.logical_or(
            env_params.god_mode,
            jnp.logical_and(
                state.item_map[
                    state.player_level, state.player_position[:, 0], state.player_position[:, 1]
                ]
                == ItemType.LADDER_DOWN.value,
                state.monsters_killed[state.player_level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL
            )
        )
    )
    is_moving_down = jnp.logical_and(
        is_moving_down,
        state.player_level < static_params.num_levels - 1
    )
    is_moving_down = is_moving_down.any()

    moving_down_position = state.up_ladders[state.player_level + 1]

    is_moving_up = jnp.logical_and(
        actions == Action.ASCEND.value,
        jnp.logical_or(
            env_params.god_mode,
            state.item_map[
                state.player_level, state.player_position[:, 0], state.player_position[:, 1]
            ]
            == ItemType.LADDER_UP.value
        )
    )
    is_moving_up = jnp.logical_and(
        is_moving_up,
        state.player_level > 0
    )
    is_moving_up = is_moving_up.any()
    
    moving_up_position = state.down_ladders[state.player_level - 1]
        
    # prioritizes moving players down levels if two players are conflicted
    position = jax.lax.select(is_moving_down, moving_down_position,
                              jax.lax.select(is_moving_up, moving_up_position, state.player_position))
    delta_floor = jax.lax.select(is_moving_down, 1,
                                 jax.lax.select(is_moving_up, -1, 0))
    
    move_down_achievement = LEVEL_ACHIEVEMENT_MAP[state.player_level + delta_floor]

    new_achievements = state.achievements.at[:, move_down_achievement].set(
        jnp.logical_or(
            (state.player_level + delta_floor) != 0,
            state.achievements[:, move_down_achievement],
        )
    )

    new_floor = jnp.logical_and(
        (state.player_level + delta_floor) != 0,
        jnp.logical_not(state.achievements[:, move_down_achievement]),
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
    def _spawn_player_projectiles(projectile_info, player_index):
        player_projectiles, player_projectile_directions, player_projectile_owners = projectile_info

        is_shooting_arrow = jnp.logical_and(
            action[player_index] == Action.SHOOT_ARROW.value,
            jnp.logical_and(
                state.inventory.bow[player_index] >= 1,
                jnp.logical_and(
                    state.inventory.arrows[player_index] >= 1,
                    player_projectiles.mask[state.player_level].sum()
                    < (static_params.max_player_projectiles * static_params.player_count),
                ),
            ),
        )

        new_player_projectiles, new_player_projectile_directions, new_player_projectile_owners = spawn_projectile(
            state, 
            static_params, 
            player_projectiles, 
            player_projectile_directions, 
            player_projectile_owners,
            state.player_position[player_index], 
            is_shooting_arrow,
            player_index,
            DIRECTIONS[state.player_direction[player_index]],
            ProjectileType.ARROW2.value,
        )

        return (new_player_projectiles, new_player_projectile_directions, new_player_projectile_owners), is_shooting_arrow

    (new_player_projectiles, new_player_projectile_directions, new_player_projectile_owners), is_shooting_arrow = jax.lax.scan(
        _spawn_player_projectiles, 
        (state.player_projectiles, state.player_projectile_directions, state.player_projectile_owners),
        jnp.arange(static_params.player_count),
    )

    new_achievements = state.achievements.at[:, Achievement.FIRE_BOW.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.FIRE_BOW.value], is_shooting_arrow
        )
    )

    return state.replace(
        player_projectiles=new_player_projectiles,
        player_projectile_directions=new_player_projectile_directions,
        player_projectile_owners=new_player_projectile_owners,
        inventory=state.inventory.replace(
            arrows=state.inventory.arrows - 1 * is_shooting_arrow
        ),
        achievements=new_achievements,
    )


def cast_spell(state, action, static_params):
    is_miner = state.player_specialization == Specialization.MINER.value
    is_warrior = state.player_specialization == Specialization.WARRIOR.value
    is_forager = state.player_specialization == Specialization.FORAGER.value

    spell_mana_cost = jnp.array([2,6]) # fireball costs 2, healing costs 5

    def _cast_player_spell(player_info, player_index):
        player_projectiles, player_projectile_directions, player_projectile_owners, player_health = player_info

        is_casting_spell = jnp.logical_and(
            action[player_index] == Action.CAST_SPELL.value,
            state.learned_spells[player_index]
        )

        # Warriors/Miners -> Cast Fireball
        is_casting_fireball = jnp.logical_and(
            is_casting_spell, state.player_mana[player_index] >= spell_mana_cost[0]
        )
        is_casting_fireball = jnp.logical_and(
            is_casting_fireball,
            jnp.logical_and(
                jnp.logical_or(is_miner[player_index], is_warrior[player_index]),
                player_projectiles.mask[state.player_level].sum()
                < (static_params.max_player_projectiles * static_params.player_count),
            )
        )
        new_player_projectiles, new_player_projectile_directions, new_player_projectile_owners = spawn_projectile(
            state,
            static_params,
            player_projectiles,
            player_projectile_directions,
            player_projectile_owners,
            state.player_position[player_index],
            is_casting_fireball,
            player_index,
            DIRECTIONS[state.player_direction[player_index]],
            ProjectileType.FIREBALL.value,
        )


        # Foragers -> Healing
        is_casting_healing = jnp.logical_and(is_casting_spell, is_forager[player_index])
        is_casting_healing = jnp.logical_and(
            is_casting_healing, state.player_mana[player_index] >= spell_mana_cost[1]
        )
        health_increase = 2
        new_player_health = jnp.minimum(
            player_health + state.player_alive * (health_increase * is_casting_healing), 
            get_max_health(state)
        )

        spell_cast = jnp.array([is_casting_fireball, is_casting_healing])

        return (new_player_projectiles, new_player_projectile_directions, new_player_projectile_owners, new_player_health), spell_cast
    
    (
        new_player_projectiles, 
        new_player_projectile_directions, 
        new_player_projectile_owners,
        new_player_health
    ), spell_cast = jax.lax.scan(
        _cast_player_spell, 
        (
            state.player_projectiles, 
            state.player_projectile_directions, 
            state.player_projectile_owners,
            state.player_health,
        ), 
        jnp.arange(static_params.player_count)
    ) 
    did_cast_spell = spell_cast.any(axis=-1)
    new_achievements = state.achievements.at[:, Achievement.CAST_SPELL.value].set(
        jnp.logical_or(state.achievements[:, Achievement.CAST_SPELL.value], did_cast_spell)
    )

    return state.replace(
        player_projectiles=new_player_projectiles,
        player_projectile_directions=new_player_projectile_directions,
        player_projectile_owners=new_player_projectile_owners,
        player_health=new_player_health,
        player_mana=state.player_mana - jnp.dot(spell_cast, spell_mana_cost),
        achievements=new_achievements,
    )


def drink_potion(state, action):
    drinking_potion_index = -1
    is_drinking_potion = False

    # Red
    is_drinking_red_potion = jnp.logical_and(
        action == Action.DRINK_POTION_RED.value, state.inventory.potions[:, 0] > 0
    )
    drinking_potion_index = (
        is_drinking_red_potion * 0
        + (1 - is_drinking_red_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_red_potion)

    # Green
    is_drinking_green_potion = jnp.logical_and(
        action == Action.DRINK_POTION_GREEN.value, state.inventory.potions[:, 1] > 0
    )
    drinking_potion_index = (
        is_drinking_green_potion * 1
        + (1 - is_drinking_green_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_green_potion)

    # Blue
    is_drinking_blue_potion = jnp.logical_and(
        action == Action.DRINK_POTION_BLUE.value, state.inventory.potions[:, 2] > 0
    )
    drinking_potion_index = (
        is_drinking_blue_potion * 2
        + (1 - is_drinking_blue_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_blue_potion)

    # Pink
    is_drinking_pink_potion = jnp.logical_and(
        action == Action.DRINK_POTION_PINK.value, state.inventory.potions[:, 3] > 0
    )
    drinking_potion_index = (
        is_drinking_pink_potion * 3
        + (1 - is_drinking_pink_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_pink_potion)

    # Cyan
    is_drinking_cyan_potion = jnp.logical_and(
        action == Action.DRINK_POTION_CYAN.value, state.inventory.potions[:, 4] > 0
    )
    drinking_potion_index = (
        is_drinking_cyan_potion * 4
        + (1 - is_drinking_cyan_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_cyan_potion)

    # Yellow
    is_drinking_yellow_potion = jnp.logical_and(
        action == Action.DRINK_POTION_YELLOW.value, state.inventory.potions[:, 5] > 0
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

    new_achievements = state.achievements.at[:, Achievement.DRINK_POTION.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.DRINK_POTION.value], is_drinking_potion
        )
    )

    return state.replace(
        inventory=state.inventory.replace(
            potions=state.inventory.potions.at[jnp.arange(state.inventory.potions.shape[0]), drinking_potion_index].set(
                state.inventory.potions[jnp.arange(state.inventory.potions.shape[0]), drinking_potion_index] - 1 * is_drinking_potion
            )
        ),
        player_health=state.player_health + delta_health,
        player_mana=state.player_mana + delta_mana,
        player_energy=state.player_energy + delta_energy,
        achievements=new_achievements,
    )


def read_book(state, action):
    is_reading_book = jnp.logical_and(
        action == Action.READ_BOOK.value, state.inventory.books > 0
    )
    new_spells = jnp.logical_or(state.learned_spells, is_reading_book)
    new_achievements = state.achievements.at[:, Achievement.LEARN_SPELL.value].set(
        jnp.logical_or(state.achievements[:, Achievement.LEARN_SPELL.value], is_reading_book)
    )

    return state.replace(
        inventory=state.inventory.replace(
            books=state.inventory.books - 1 * is_reading_book
        ),
        learned_spells=new_spells,
        achievements=new_achievements,
    )


def enchant(rng, state: EnvState, action, static_params: StaticEnvParams):
    target_block_position = state.player_position + DIRECTIONS[state.player_direction]
    target_block = state.map[
        state.player_level, target_block_position[:, 0], target_block_position[:, 1]
    ]
    target_block_is_enchantment_table = jnp.logical_or(
        target_block == BlockType.ENCHANTMENT_TABLE_FIRE.value,
        target_block == BlockType.ENCHANTMENT_TABLE_ICE.value,
    )

    enchantment_type = jnp.where(
        target_block == BlockType.ENCHANTMENT_TABLE_FIRE.value, 
        1, 
        2
    )

    num_gems = jnp.where(
        target_block == BlockType.ENCHANTMENT_TABLE_FIRE.value,
        state.inventory.ruby,
        state.inventory.sapphire,
    )

    could_enchant = jnp.logical_and(
        state.player_mana >= 9,
        jnp.logical_and(target_block_is_enchantment_table, num_gems >= 1),
    )
    could_enchant_warrior = jnp.logical_and(
        state.player_specialization == Specialization.WARRIOR.value,
        could_enchant
    )

    is_enchanting_bow = jnp.logical_and(
        could_enchant_warrior,
        jnp.logical_and(action == Action.ENCHANT_BOW.value, state.inventory.bow > 0),
    )

    is_enchanting_sword = jnp.logical_and(
        could_enchant_warrior,
        jnp.logical_and(
            action == Action.ENCHANT_SWORD.value, state.inventory.sword > 0
        ),
    )

    is_enchanting_armour = jnp.logical_and(
        could_enchant,
        jnp.logical_and(
            action == Action.ENCHANT_ARMOUR.value, state.inventory.armour.sum(axis=1) > 0
        ),
    )

    rng, _rng = jax.random.split(rng)
    unenchanted_armour = state.armour_enchantments == 0
    opposite_enchanted_armour = jnp.logical_and(
        state.armour_enchantments != 0, state.armour_enchantments != enchantment_type
    )

    armour_targets = (
        unenchanted_armour + (unenchanted_armour.sum(axis=1) == 0) * opposite_enchanted_armour
    )

    _rngs = jax.random.split(rng, static_params.player_count+1)
    rng, _rng = _rngs[0], _rngs[1:]
    armour_target = jax.vmap(jax.random.choice, in_axes=(0, None, None, None, 0))(
        _rng, jnp.arange(4), (), True, armour_targets
    )

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

    new_armour_enchantments = state.armour_enchantments.at[jnp.arange(static_params.player_count), armour_target].set(
        is_enchanting_armour * enchantment_type
        + (1 - is_enchanting_armour) * state.armour_enchantments[jnp.arange(static_params.player_count), armour_target]
    )

    new_sapphire = state.inventory.sapphire - 1 * is_enchanting * (
        enchantment_type == 2
    )
    new_ruby = state.inventory.ruby - 1 * is_enchanting * (enchantment_type == 1)
    new_mana = state.player_mana - 9 * is_enchanting

    new_achievements = state.achievements.at[:, Achievement.ENCHANT_SWORD.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.ENCHANT_SWORD.value], is_enchanting_sword
        )
    )

    new_achievements = new_achievements.at[:, Achievement.ENCHANT_ARMOUR.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.ENCHANT_ARMOUR.value], is_enchanting_armour
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
    new_achievements = state.achievements.at[:, Achievement.DEFEAT_NECROMANCER.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.DEFEAT_NECROMANCER.value],
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
    achievements = state.achievements.at[:, Achievement.COLLECT_WOOD.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.COLLECT_WOOD.value], state.inventory.wood > 0
        )
    )
    # Stone
    achievements = achievements.at[:, Achievement.COLLECT_STONE.value].set(
        jnp.logical_or(
            achievements[:, Achievement.COLLECT_STONE.value], state.inventory.stone > 0
        )
    )
    # Coal
    achievements = achievements.at[:, Achievement.COLLECT_COAL.value].set(
        jnp.logical_or(
            achievements[:, Achievement.COLLECT_COAL.value], state.inventory.coal > 0
        )
    )
    # Iron
    achievements = achievements.at[:, Achievement.COLLECT_IRON.value].set(
        jnp.logical_or(
            achievements[:, Achievement.COLLECT_IRON.value], state.inventory.iron > 0
        )
    )
    # Diamond
    achievements = achievements.at[:, Achievement.COLLECT_DIAMOND.value].set(
        jnp.logical_or(
            achievements[:, Achievement.COLLECT_DIAMOND.value], state.inventory.diamond > 0
        )
    )
    # Ruby
    achievements = achievements.at[:, Achievement.COLLECT_RUBY.value].set(
        jnp.logical_or(
            achievements[:, Achievement.COLLECT_RUBY.value], state.inventory.ruby > 0
        )
    )
    # Sapphire
    achievements = achievements.at[:, Achievement.COLLECT_SAPPHIRE.value].set(
        jnp.logical_or(
            achievements[:, Achievement.COLLECT_SAPPHIRE.value],
            state.inventory.sapphire > 0,
        )
    )
    # Sapling
    achievements = achievements.at[:, Achievement.COLLECT_SAPLING.value].set(
        jnp.logical_or(
            achievements[:, Achievement.COLLECT_SAPLING.value], state.inventory.sapling > 0
        )
    )
    # Bow
    achievements = achievements.at[:, Achievement.FIND_BOW.value].set(
        jnp.logical_or(
            achievements[:, Achievement.FIND_BOW.value], state.inventory.bow > 0
        )
    )
    # Arrow
    achievements = achievements.at[:, Achievement.MAKE_ARROW.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_ARROW.value], state.inventory.arrows > 0
        )
    )
    # Torch
    achievements = achievements.at[:, Achievement.MAKE_TORCH.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_TORCH.value], state.inventory.torches > 0
        )
    )

    # Pickaxe
    achievements = achievements.at[:, Achievement.MAKE_WOOD_PICKAXE.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_WOOD_PICKAXE.value],
            state.inventory.pickaxe >= 1,
        )
    )
    achievements = achievements.at[:, Achievement.MAKE_STONE_PICKAXE.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_STONE_PICKAXE.value],
            state.inventory.pickaxe >= 2,
        )
    )
    achievements = achievements.at[:, Achievement.MAKE_IRON_PICKAXE.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_IRON_PICKAXE.value],
            state.inventory.pickaxe >= 3,
        )
    )
    achievements = achievements.at[:, Achievement.MAKE_DIAMOND_PICKAXE.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_DIAMOND_PICKAXE.value],
            state.inventory.pickaxe >= 4,
        )
    )

    # Sword
    achievements = achievements.at[:, Achievement.MAKE_WOOD_SWORD.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_WOOD_SWORD.value], state.inventory.sword >= 1
        )
    )
    achievements = achievements.at[:, Achievement.MAKE_STONE_SWORD.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_STONE_SWORD.value], state.inventory.sword >= 2
        )
    )
    achievements = achievements.at[:, Achievement.MAKE_IRON_SWORD.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_IRON_SWORD.value], state.inventory.sword >= 3
        )
    )
    achievements = achievements.at[:, Achievement.MAKE_DIAMOND_SWORD.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_DIAMOND_SWORD.value],
            state.inventory.sword >= 4,
        )
    )

    return state.replace(achievements=achievements)


def trade_materials(state, action, static_params):
    new_achievements = state.achievements
    

    player_trading_to = action - Action.GIVE.value
    is_giving = jnp.logical_and(
        jnp.logical_and(
            action >= Action.GIVE.value, 
            action < (Action.GIVE.value + static_params.player_count)
        ),
        player_trading_to != jnp.arange(static_params.player_count) # isn't giving to self 
    )
    other_player_is_requesting = jnp.logical_and(
        state.request_duration[player_trading_to] > 0,
        state.player_alive[player_trading_to]
    )

    def _new_material_value(material_type, current_material_stock, material_max_value):
        other_player_is_requesting_material = jnp.logical_and(
            other_player_is_requesting,
            state.request_type[player_trading_to] == material_type
        )
        is_giving_material = jnp.logical_and(
            jnp.logical_and( # Checks that other player is requesting and can take materials
                other_player_is_requesting_material,
                current_material_stock[player_trading_to] < material_max_value
            ),
            jnp.logical_and( # Checks that player has materials and is giving
                is_giving,
                current_material_stock > 0
            )
        )
        new_material = current_material_stock - 1 * is_giving_material
        new_material = new_material.at[player_trading_to].add(is_giving_material)
        return new_material
    
    # Food
    new_food = _new_material_value(
        Action.REQUEST_FOOD.value, state.player_food, get_max_food(state)
    )
    new_hunger = jnp.where(new_food>state.player_food, 0.0, state.player_hunger)
    new_achievements = new_achievements.at[:, Achievement.COLLECT_FOOD.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.COLLECT_FOOD.value], new_food>state.player_food
        )
    )
    
    # Drink
    new_drink = _new_material_value(
        Action.REQUEST_DRINK.value, state.player_drink, get_max_drink(state)
    )
    new_thirst = jnp.where(new_drink>state.player_drink, 0.0, state.player_thirst)
    new_achievements = new_achievements.at[:, Achievement.COLLECT_DRINK.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.COLLECT_DRINK.value], new_drink>state.player_drink
        )
    )

    # Inventory Materials
    new_wood = _new_material_value(
        Action.REQUEST_WOOD.value, state.inventory.wood, 99
    )
    new_stone = _new_material_value(
        Action.REQUEST_STONE.value, state.inventory.stone, 99
    )
    new_iron = _new_material_value(
        Action.REQUEST_IRON.value, state.inventory.iron, 99
    )
    new_coal = _new_material_value(
        Action.REQUEST_COAL.value, state.inventory.coal, 99
    )
    new_diamond = _new_material_value(
        Action.REQUEST_DIAMOND.value, state.inventory.diamond, 99
    )
    new_ruby = _new_material_value(
        Action.REQUEST_RUBY.value, state.inventory.ruby, 99
    )
    new_sapphire = _new_material_value(
        Action.REQUEST_SAPPHIRE.value, state.inventory.sapphire, 99
    )

    # Update State
    state = state.replace(
        player_food=new_food,
        player_drink=new_drink,
        player_hunger=new_hunger,
        player_thirst=new_thirst,
        inventory=state.inventory.replace(
            wood=new_wood,
            stone=new_stone,
            iron=new_iron,
            coal=new_coal,
            diamond=new_diamond,
            ruby=new_ruby,
            sapphire=new_sapphire,
        ),
        achievements=new_achievements,
    )
    return state


def make_request(state, action):
    # Decrease Request Duration
    state = state.replace(
        request_duration=jnp.maximum(0, state.request_duration - 1)
    )

    # Initialize New Request
    is_making_request = jnp.logical_and( # Hacky
        action >= Action.REQUEST_FOOD.value, action <= Action.REQUEST_DIAMOND.value
    )
    new_request_type = jnp.where(
        is_making_request,
        action,
        state.request_type
    )
    state = state.replace(
        request_duration=jnp.maximum(state.request_duration, is_making_request * REQUEST_MAX_DURATION),
        request_type=new_request_type
    )
    return state


def level_up_attributes(state: EnvState, action: jnp.array, params: EnvParams) -> EnvState:
    can_level_up = state.player_xp >= 1

    # Specializing
    can_specialize = (state.player_specialization == Specialization.UNASSIGNED.value)
    new_specialization = can_specialize * (
        (action == Action.SELECT_FORAGER.value) * Specialization.FORAGER.value +
        (action == Action.SELECT_WARRIOR.value) * Specialization.WARRIOR.value +
        (action == Action.SELECT_MINER.value) * Specialization.MINER.value
    ) + (1-can_specialize) * state.player_specialization

    # Levelling up attributes
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
        player_specialization=new_specialization,
        player_xp=state.player_xp - 1 * is_levelling_up,
    )


def craftax_step(
        rng: chex.PRNGKey, state: EnvState, actions: jnp.array, params: EnvParams, static_params: StaticEnvParams
    ) -> Tuple[EnvState, chex.Array]:
    init_achievements = state.achievements
    init_health = state.player_health

    # Interrupt action if dead, sleeping or resting
    cant_do_action = jnp.logical_or(
        jnp.logical_not(state.player_alive),
        jnp.logical_or(state.is_sleeping, state.is_resting),
    )
    actions = jnp.where(
        cant_do_action,
        Action.NOOP.value,
        actions
    )

    # Change floor
    state = change_floor(state, actions, params, static_params)

    # Crafting
    state = do_crafting(state, actions, static_params)

    # Interact (mining, melee attacking, eating plants, drinking water, reviving)
    rng, _rng = jax.random.split(rng)
    state = do_action(_rng, state, actions, params, static_params)

    # Placing
    state = place_block(state, actions, static_params)

    # Shooting
    state = shoot_projectile(state, actions, static_params)

    # Casting
    state = cast_spell(state, actions, static_params)

    # Potions
    state = drink_potion(state, actions)

    # Read
    state = read_book(state, actions)

    # Enchant
    rng, _rng = jax.random.split(rng)
    state = enchant(_rng, state, actions, static_params)

    # Boss
    state = boss_logic(state, static_params)

    # Attributes
    state = level_up_attributes(state, actions, params)

    # Trade
    state = trade_materials(state, actions, static_params)

    # Request Materials
    state = make_request(state, actions)

    # Movement
    state = move_player(state, actions, params, static_params)

    # Mobs
    rng, _rng = jax.random.split(rng)
    state = update_mobs(_rng, state, params, params, static_params)

    rng, _rng = jax.random.split(rng)
    state = spawn_mobs(state, _rng, params, static_params)

    # Plants
    state = update_plants(state, static_params)

    # Intrinsics
    state = update_player_intrinsics(state, actions, static_params)

    # Cap inv
    state = clip_inventory_and_intrinsics(state, params)

    # Inventory achievements
    state = calculate_inventory_achievements(state)

    # Reward
    achievement_coefficients = ACHIEVEMENT_REWARD_MAP
    achievement_reward = (
        (state.achievements.astype(int) - init_achievements.astype(int))
        * achievement_coefficients
    ).sum(axis=1)

    # Gain reward if player gained health
    # Doesn't apply to revived players
    health_reward = jnp.where(
        state.player_alive,
        (state.player_health - init_health) * 0.1,
        0.0
    )

    individual_reward = achievement_reward + health_reward
    shared_reward = individual_reward.sum().repeat(static_params.player_count)

    reward = jax.lax.select(
        params.shared_reward,
        shared_reward,
        individual_reward
    )

    player_alive = state.player_health > 0.0

    rng, _rng = jax.random.split(rng)

    state = state.replace(
        player_alive=player_alive,
        timestep=state.timestep + 1,
        light_level=calculate_light_level(state.timestep + 1, params),
        state_rng=_rng,
    )

    return state, reward
