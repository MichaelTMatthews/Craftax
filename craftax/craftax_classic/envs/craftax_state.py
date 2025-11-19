from dataclasses import dataclass
from typing import Tuple, Any

import jax.random
from flax import struct
import jax.numpy as jnp

@struct.dataclass
class Inventory:
    wood: int = 0
    stone: int = 0
    coal: int = 0
    iron: int = 0
    diamond: int = 0
    sapling: int = 0
    wood_pickaxe: int = 0
    stone_pickaxe: int = 0
    iron_pickaxe: int = 0
    wood_sword: int = 0
    stone_sword: int = 0
    iron_sword: int = 0

@struct.dataclass
class Mobs:
    position: jnp.ndarray
    health: int
    mask: bool
    attack_cooldown: int

@struct.dataclass
class EnvState:
    map: jnp.ndarray
    mob_map: jnp.ndarray

    player_position: jnp.ndarray
    player_direction: int
    default_reward: float

    # Intrinsics
    player_health: int
    player_food: int
    player_drink: int
    player_energy: int
    player_accomplishment: float
    is_sleeping: bool


    player_uHealth: float

    player_uFood: float
    player_uDrink: float
    player_uEnergy: float
    player_uAccomplishment: float

    player_uHealth_th: float
    player_uFood_th: float
    player_uDrink_th: float
    player_uEnergy_th: float
    player_uAccomplishment_th: float

    player_sHealth: float
    player_sFood: float
    player_sDrink: float
    player_sEnergy: float
    player_sAccomplishment: float

    # Second order intrinsics
    player_recover: float
    player_hunger: float
    player_thirst: float
    player_fatigue: float

    inventory: Inventory

    zombies: Mobs
    cows: Mobs
    skeletons: Mobs
    arrows: Mobs
    arrow_directions: jnp.ndarray

    growing_plants_positions: jnp.ndarray
    growing_plants_age: jnp.ndarray
    growing_plants_mask: jnp.ndarray

    light_level: float

    achievements: jnp.ndarray

    state_rng: Any

    timestep: int

    fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)


@struct.dataclass
class EnvParams:
    max_timesteps: int = 10000
    day_length: int = 300

    always_diamond: bool = True

    zombie_health: int = 5
    cow_health: int = 3
    skeleton_health: int = 3

    mob_despawn_distance: int = 14

    spawn_cow_chance: float = 0.1
    spawn_zombie_base_chance: float = 0.02
    spawn_zombie_night_chance: float = 0.1
    spawn_skeleton_chance: float = 0.05

    fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)


@struct.dataclass
class StaticEnvParams:
    map_size: Tuple[int, int] = (64, 64)

    # Mobs
    max_zombies: int = 3
    max_cows: int = 3
    max_growing_plants: int = 10
    max_skeletons: int = 2
    max_arrows: int = 3
