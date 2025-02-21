import chex
import jax
from functools import partial
from jax import lax
from typing import Dict, Tuple
from jaxmarl.environments import spaces
from jaxmarl.environments.multi_agent_env import MultiAgentEnv

from craftax_marl.constants import *
from craftax_marl.craftax_state import EnvState, EnvParams, StaticEnvParams
from craftax_marl.envs.common import compute_score
from craftax_marl.game_logic import craftax_step
from craftax_marl.renderer.renderer_symbolic import render_craftax_symbolic
from craftax_marl.util.game_logic_utils import has_beaten_boss
from craftax_marl.world_gen.world_gen import generate_world


class CraftaxMARLSymbolicEnv(MultiAgentEnv):
    def __init__(self, num_agents: int = 3):
        self.num_agents = num_agents
        self.static_env_params = CraftaxMARLSymbolicEnv.default_static_params()

        self.player_names = [
            f"agent_{i}" for i in range(self.static_env_params.player_count)
        ]

        self.action_spaces = {name: self.action_shape() for name in self.player_names}
        self.observation_spaces = {name: self.observation_shape() for name in self.player_names}

        self.player_names = [
            f"agent_{i}" for i in range(self.static_env_params.player_count)
        ]

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        state = generate_world(key, self.default_params, self.static_env_params)
        return self.get_obs(state), state
    
    def step_env(
        self, key: chex.PRNGKey, state: EnvState, actions: Dict[str, chex.Array]
    ) -> Tuple[chex.Array, EnvState, Dict[str, float], Dict[str, bool], Dict]:
        actions = jnp.array(list(actions.values()))
        state, reward = craftax_step(key, state, actions, self.default_params, self.static_env_params)

        obs = self.get_obs(state)
        done = self.is_terminal(state, self.default_params)
        info = compute_score(state, done)
        info["discount"] = self.discount(state, self.default_params)

        agent_rewards = {n: r for n,r in zip(self.player_names, reward)}

        agent_done = {n: done for n in self.player_names}
        agent_done["__all__"] = done

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            agent_rewards,
            agent_done,
            info,
        )

    def get_obs(self, state: EnvState) -> chex.Array:
        obs_sym = render_craftax_symbolic(
            state, 
            self.static_env_params,
        )
        return obs_sym
    
    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
    
    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()
    
    def action_shape(self) -> spaces.Discrete:
        return spaces.Discrete(len(Action) + (self.static_env_params.player_count - 1))
    
    def get_flat_map_obs_shape(self):
        num_mob_classes = 5
        num_mob_types = 8
        num_blocks = len(BlockType)
        num_items = len(ItemType)
        num_players = self.static_env_params.player_count
        teammate_dead_alive_bit = 1
        light_map = 1

        return (
            OBS_DIM[0] *
            OBS_DIM[1] *
            (num_players + teammate_dead_alive_bit + num_blocks + num_items + num_mob_classes * num_mob_types + light_map)
        )

    def get_teammate_dashboard_obs_shape(self):
        num_players = self.static_env_params.player_count
        num_health = 1
        num_alive = 1
        num_specialization = len(Specialization) - 1
        num_req_mats = (Action.REQUEST_SAPPHIRE.value - Action.REQUEST_FOOD.value + 1)
        num_directions = 8

        return num_players * (num_health + num_alive + num_specialization + num_req_mats + num_directions)

    def get_inventory_obs_shape(self):
        num_inventory = 16
        num_potions = 6
        num_intrinsics = 8
        num_directions = 4
        num_armour = 4
        num_armour_enchantments = 4
        num_special_values = 3
        num_special_level_values = 4
        return num_inventory + num_potions + num_intrinsics + num_directions + num_armour + num_armour_enchantments + num_special_values + num_special_level_values
    
    def observation_shape(self) -> spaces.Box:
        flat_map_obs_shape = self.get_flat_map_obs_shape()
        teammate_dashboard_obs_shape = self.get_teammate_dashboard_obs_shape()
        inventory_obs_shape = self.get_inventory_obs_shape()
        obs_shape = flat_map_obs_shape + teammate_dashboard_obs_shape + inventory_obs_shape

        return spaces.Box(
            0.0,
            1.0,
            (obs_shape,),
            dtype=jnp.int32,
        )
    
    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        done_steps = state.timestep >= params.max_timesteps
        is_dead = jnp.logical_not(state.player_alive).all()
        defeated_boss = has_beaten_boss(state, self.static_env_params)
        is_terminal = jnp.logical_or(is_dead, done_steps)
        is_terminal = jnp.logical_or(is_terminal, defeated_boss)
        return is_terminal
    
    def discount(self, state, params) -> float:
        """Return a discount of zero if the episode has terminated."""
        return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)
