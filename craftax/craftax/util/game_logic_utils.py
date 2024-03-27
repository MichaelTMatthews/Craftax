import chex

from craftax.craftax.constants import *
from craftax.craftax.craftax_state import *

# For utility functions - functions called more than once in meaningfully different parts of the codebase


def is_fighting_boss(state, static_params):
    return state.player_level == (static_params.num_levels - 1)


def is_boss_spawn_wave(state, static_params):
    return jnp.logical_and(
        is_fighting_boss(state, static_params),
        state.boss_timesteps_to_spawn_this_round >= 1,
    )


def is_boss_vulnerable(state):
    return jnp.logical_and(
        state.melee_mobs.mask[state.player_level].sum() == 0,
        jnp.logical_and(
            state.ranged_mobs.mask[state.player_level].sum() == 0,
            state.boss_timesteps_to_spawn_this_round <= 0,
        ),
    )


def has_beaten_boss(state, static_params):
    return state.boss_progress >= static_params.num_levels - 1


def attack_mob_class(
    state,
    mobs,
    position,
    damage_vector,
    can_get_achievement,
    mob_class_index,
):
    def is_attacking_mob_at_index(unused, mob_index):
        in_mob = (mobs.position[state.player_level, mob_index] == position).all()
        return None, jnp.logical_and(in_mob, mobs.mask[state.player_level, mob_index])

    _, is_attacking_mob_array = jax.lax.scan(
        is_attacking_mob_at_index, None, jnp.arange(mobs.mask.shape[1])
    )
    is_attacking_mob = is_attacking_mob_array.sum() > 0
    target_mob_index = jnp.argmax(is_attacking_mob_array)

    damage = get_damage(
        damage_vector,
        MOB_TYPE_DEFENSE_MAPPING[
            mobs.type_id[state.player_level, target_mob_index], mob_class_index
        ],
    )

    new_mob_health = mobs.health.at[state.player_level, target_mob_index].add(
        -damage * is_attacking_mob
    )
    mobs = mobs.replace(health=new_mob_health)

    old_mask = mobs.mask[state.player_level, target_mob_index]
    mobs = mobs.replace(mask=jnp.logical_and(mobs.health > 0, mobs.mask))
    did_kill_mob = jnp.logical_and(
        old_mask,
        jnp.logical_not(mobs.mask[state.player_level, target_mob_index]),
    )

    achievement_for_kill = MOB_ACHIEVEMENT_MAP[
        mob_class_index, mobs.type_id[state.player_level, target_mob_index]
    ]

    new_achievements = state.achievements.at[achievement_for_kill].set(
        jnp.logical_or(
            state.achievements[achievement_for_kill],
            jnp.logical_and(did_kill_mob, can_get_achievement),
        )
    )

    return mobs, did_kill_mob, is_attacking_mob, new_achievements


def attack_mob(state, position, damage_vector, can_eat):
    # Melee
    (
        new_melee_mobs,
        did_kill_melee_mob,
        is_attacking_melee_mob,
        new_achievements,
    ) = attack_mob_class(
        state,
        state.melee_mobs,
        position,
        damage_vector,
        True,
        1,
    )

    state = state.replace(
        melee_mobs=new_melee_mobs,
        achievements=new_achievements,
    )

    # Cow
    (
        new_passive_mobs,
        did_kill_passive_mob,
        is_attacking_passive_mob,
        new_achievements,
    ) = attack_mob_class(
        state,
        state.passive_mobs,
        position,
        damage_vector,
        can_eat,
        0,
    )

    new_food = jax.lax.select(
        jnp.logical_and(did_kill_passive_mob, can_eat),
        jnp.minimum(get_max_food(state), state.player_food + 6),
        state.player_food,
    )
    new_hunger = jax.lax.select(
        jnp.logical_and(did_kill_passive_mob, can_eat), 0.0, state.player_hunger
    )

    state = state.replace(
        passive_mobs=new_passive_mobs,
        player_food=new_food,
        player_hunger=new_hunger,
        achievements=new_achievements,
    )

    # Skeleton
    (
        new_ranged_mobs,
        did_kill_ranged_mob,
        is_attacking_ranged_mob,
        new_achievements,
    ) = attack_mob_class(
        state,
        state.ranged_mobs,
        position,
        damage_vector,
        True,
        2,
    )

    state = state.replace(
        ranged_mobs=new_ranged_mobs,
        achievements=new_achievements,
    )

    # Update mob map on kill

    did_attack_mob = jnp.logical_or(
        jnp.logical_or(is_attacking_melee_mob, is_attacking_passive_mob),
        is_attacking_ranged_mob,
    )

    did_kill_monster = jnp.logical_or(did_kill_melee_mob, did_kill_ranged_mob)
    did_kill_mob = jnp.logical_or(did_kill_monster, did_kill_passive_mob)

    state = state.replace(
        mob_map=state.mob_map.at[state.player_level, position[0], position[1]].set(
            jnp.logical_and(
                state.mob_map[state.player_level, position[0], position[1]],
                jnp.logical_not(did_kill_mob),
            )
        ),
        monsters_killed=state.monsters_killed.at[state.player_level].add(
            1 * did_kill_monster
        ),
    )

    return state, did_attack_mob, did_kill_mob


def spawn_projectile(
    state,
    static_params,
    projectiles,
    projectile_directions,
    new_projectile_position,
    is_spawning_projectile,
    direction,
    projectile_type,
):
    new_projectile_index = jnp.argmax(
        jnp.logical_not(projectiles.mask[state.player_level])
    )
    new_projectile_position = jax.lax.select(
        is_spawning_projectile,
        new_projectile_position,
        projectiles.position[state.player_level, new_projectile_index],
    )
    new_projectile_mask = jax.lax.select(
        is_spawning_projectile,
        True,
        projectiles.mask[state.player_level, new_projectile_index],
    )
    new_projectile_direction = jax.lax.select(
        is_spawning_projectile,
        direction,
        projectile_directions[state.player_level, new_projectile_index],
    )
    new_projectile_type = jax.lax.select(
        is_spawning_projectile,
        projectile_type,
        projectiles.type_id[state.player_level, new_projectile_index],
    )

    new_projectiles = projectiles.replace(
        position=projectiles.position.at[state.player_level, new_projectile_index].set(
            new_projectile_position
        ),
        mask=projectiles.mask.at[state.player_level, new_projectile_index].set(
            new_projectile_mask
        ),
        type_id=projectiles.type_id.at[state.player_level, new_projectile_index].set(
            new_projectile_type
        ),
    )

    new_projectile_directions = projectile_directions.at[
        state.player_level, new_projectile_index
    ].set(new_projectile_direction)

    return new_projectiles, new_projectile_directions


def get_damage_done_to_player(state, static_params, damage_vector):
    scaled_defenses = jnp.stack(
        [
            state.inventory.armour * 0.1,
            (state.armour_enchantments == 1) * 0.2,
            (state.armour_enchantments == 2) * 0.2,
        ],
        axis=0,
    )

    defense_vector = scaled_defenses.sum(axis=1)

    damage_vector *= (
        1 + is_fighting_boss(state, static_params) * BOSS_FIGHT_EXTRA_DAMAGE
    )

    return get_damage(damage_vector, defense_vector)


def get_player_damage_vector(state):
    physical_damages = jnp.array(
        [1, 2, 3, 5, 8],
        dtype=jnp.int32,
    )
    physical_damage = physical_damages[state.inventory.sword]
    fire_damage = physical_damage * (state.sword_enchantment == 1) * 0.5
    ice_damage = physical_damage * (state.sword_enchantment == 2) * 0.5

    physical_damage *= 1 + 0.25 * (
        state.player_strength - 1
    )  # Strength=5 does double damage
    fire_damage *= 1 + 0.05 * (
        state.player_intelligence - 1
    )  # Int=5 does 25% more enchant damage
    ice_damage *= 1 + 0.05 * (
        state.player_intelligence - 1
    )  # Int=5 does 25% more enchant damage

    return jnp.stack([physical_damage, fire_damage, ice_damage], axis=0)


def get_damage(damage_vector, defense_vector):
    damages = (1.0 - defense_vector) * damage_vector

    return damages.sum()


def in_bounds(state, position):
    in_bounds_x = jnp.logical_and(
        0 <= position[0], position[0] < state.map[state.player_level].shape[0]
    )
    in_bounds_y = jnp.logical_and(
        0 <= position[1], position[1] < state.map[state.player_level].shape[1]
    )
    return jnp.logical_and(in_bounds_x, in_bounds_y)


def is_in_solid_block(state, position):
    return SOLID_BLOCK_MAPPING[state.map[state.player_level, position[0], position[1]]]


def is_position_in_bounds_not_in_mob_not_colliding(state, position, collision_map):
    pos_in_bounds = in_bounds(state, position)
    in_solid_block = is_in_solid_block(state, position)
    in_mob = is_in_mob(state, position)
    in_lava = (
        state.map[state.player_level][position[0], position[1]] == BlockType.LAVA.value
    )
    in_water = (
        state.map[state.player_level][position[0], position[1]] == BlockType.WATER.value
    )
    on_ground_block = jnp.logical_and(
        jnp.logical_not(in_solid_block),
        jnp.logical_and(jnp.logical_not(in_water), jnp.logical_not(in_lava)),
    )

    valid_move = jnp.logical_and(
        pos_in_bounds,
        jnp.logical_and(jnp.logical_not(in_mob), jnp.logical_not(in_solid_block)),
    )

    # Ground blocks
    valid_move = jnp.logical_and(
        valid_move,
        jnp.logical_or(
            jnp.logical_not(collision_map[0]), jnp.logical_not(on_ground_block)
        ),
    )

    # Water
    valid_move = jnp.logical_and(
        valid_move,
        jnp.logical_or(jnp.logical_not(collision_map[1]), jnp.logical_not(in_water)),
    )

    # Lava
    valid_move = jnp.logical_and(
        valid_move,
        jnp.logical_or(jnp.logical_not(collision_map[2]), jnp.logical_not(in_lava)),
    )

    return valid_move


def is_near_block(state, block_type):
    def _is_given_block(unused, loc_add):
        pos = state.player_position + loc_add
        is_in_bounds = in_bounds(state, pos)
        is_correct_block = state.map[state.player_level][pos[0], pos[1]] == block_type
        return None, jnp.logical_and(is_in_bounds, is_correct_block)

    _, is_block = jax.lax.scan(_is_given_block, None, CLOSE_BLOCKS)

    return is_block.sum() > 0


def calculate_light_level(timestep, params):
    progress = (timestep / params.day_length) % 1 + 0.3
    return 1 - jnp.abs(jnp.cos(jnp.pi * progress)) ** 3


def is_in_mob(state: EnvState, position: chex.Array):
    return jnp.logical_or(
        state.mob_map[state.player_level, position[0], position[1]],
        (state.player_position == position).all(),
    )


def get_max_health(state):
    return 8 + state.player_strength


def get_max_food(state):
    return 7 + 2 * state.player_dexterity


def get_max_drink(state):
    return 7 + 2 * state.player_dexterity


def get_max_energy(state):
    return 7 + 2 * state.player_dexterity


def get_max_mana(state):
    return 6 + 3 * state.player_intelligence


def clip_inventory_and_intrinsics(state, params):
    capped_inv = jax.tree_map(lambda x: jnp.minimum(x, 99), state.inventory)

    min_health = jax.lax.select(params.god_mode, 9, 0)

    state = state.replace(
        inventory=capped_inv,
        player_health=jnp.minimum(
            jnp.maximum(state.player_health, min_health), get_max_health(state)
        ),
        player_food=jnp.minimum(jnp.maximum(state.player_food, 0), get_max_food(state)),
        player_drink=jnp.minimum(
            jnp.maximum(state.player_drink, 0), get_max_drink(state)
        ),
        player_energy=jnp.minimum(
            jnp.maximum(state.player_energy, 0), get_max_energy(state)
        ),
        player_mana=jnp.minimum(jnp.maximum(state.player_mana, 0), get_max_mana(state)),
    )

    return state
