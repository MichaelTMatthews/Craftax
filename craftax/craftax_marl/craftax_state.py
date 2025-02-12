from dataclasses import dataclass
from typing import Tuple, Any

import jax
from flax import struct
import jax.numpy as jnp


@struct.dataclass
class Inventory:
    wood: jnp.ndarray
    stone: jnp.ndarray
    coal: jnp.ndarray
    iron: jnp.ndarray
    diamond: jnp.ndarray
    sapling: jnp.ndarray
    pickaxe: jnp.ndarray
    sword: jnp.ndarray
    bow: jnp.ndarray
    arrows: jnp.ndarray
    armour: jnp.ndarray
    torches: jnp.ndarray
    ruby: jnp.ndarray
    sapphire: jnp.ndarray
    potions: jnp.ndarray
    books: jnp.ndarray


@struct.dataclass
class Mobs:
    position: jnp.ndarray
    health: jnp.ndarray
    mask: jnp.ndarray
    attack_cooldown: jnp.ndarray
    type_id: jnp.ndarray


# @struct.dataclass
# class Projectiles(Mobs):
#     directions: jnp.ndarray
#     lifetimes: jnp.ndarray


@struct.dataclass
class EnvState:
    map: jnp.ndarray
    item_map: jnp.ndarray
    mob_map: jnp.ndarray
    light_map: jnp.ndarray
    down_ladders: jnp.ndarray
    up_ladders: jnp.ndarray
    chests_opened: jnp.ndarray
    monsters_killed: jnp.ndarray

    player_position: jnp.ndarray
    player_level: int
    player_direction: jnp.ndarray
    player_alive: jnp.ndarray

    # Intrinsics
    player_health: jnp.ndarray
    player_food: jnp.ndarray
    player_drink: jnp.ndarray
    player_energy: jnp.ndarray
    player_mana: jnp.ndarray
    is_sleeping: jnp.ndarray
    is_resting: jnp.ndarray

    # Second order intrinsics
    player_recover: jnp.ndarray
    player_hunger: jnp.ndarray
    player_thirst: jnp.ndarray
    player_fatigue: jnp.ndarray
    player_recover_mana: jnp.ndarray

    # Attributes
    player_xp: jnp.ndarray
    player_dexterity: jnp.ndarray
    player_strength: jnp.ndarray
    player_intelligence: jnp.ndarray
    player_specialization: jnp.ndarray

    # Request Info
    request_duration: jnp.ndarray
    request_type: jnp.ndarray

    inventory: Inventory

    melee_mobs: Mobs
    passive_mobs: Mobs
    ranged_mobs: Mobs

    mob_projectiles: Mobs
    mob_projectile_directions: jnp.ndarray
    mob_projectile_owners: jnp.ndarray
    player_projectiles: Mobs
    player_projectile_directions: jnp.ndarray
    player_projectile_owners: jnp.ndarray

    growing_plants_positions: jnp.ndarray
    growing_plants_age: jnp.ndarray
    growing_plants_mask: jnp.ndarray

    potion_mapping: jnp.ndarray
    learned_spells: jnp.ndarray

    sword_enchantment: jnp.ndarray
    bow_enchantment: jnp.ndarray
    armour_enchantments: jnp.ndarray

    boss_progress: int
    boss_timesteps_to_spawn_this_round: int

    light_level: float

    achievements: jnp.ndarray

    state_rng: Any

    timestep: int

    fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)


@struct.dataclass
class EnvParams:
    max_timesteps: int = 100000
    day_length: int = 300

    melee_mob_health: int = 5
    passive_mob_health: int = 3
    ranged_mob_health: int = 3

    mob_despawn_distance: int = 14
    max_attribute: int = 5


    fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)

    # Game Mode Parameters
    god_mode: bool = False
    shared_reward: bool = True
    friendly_fire: bool = True

@struct.dataclass
class StaticEnvParams:
    map_size: Tuple[int, int] = (48, 48)
    num_levels: int = 9
    player_count: int = 3

    # Mobs Per Player
    max_melee_mobs: int = 3
    max_passive_mobs: int = 3
    max_growing_plants: int = 10
    max_ranged_mobs: int = 2
    max_mob_projectiles: int = 3
    max_player_projectiles: int = 3
