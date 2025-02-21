import chex
import jax
from functools import partial
from jax import lax
from jaxmarl.environments import spaces
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from typing import Dict, Tuple

from craftax_marl.constants import *
from craftax_marl.craftax_state import EnvState, EnvParams, StaticEnvParams
from craftax_marl.envs.common import compute_score
from craftax_marl.game_logic import craftax_step
from craftax_marl.renderer.renderer_pixels import render_craftax_pixels
from craftax_marl.util.game_logic_utils import has_beaten_boss
from craftax_marl.world_gen.world_gen import generate_world


class CraftaxMARLPixelsEnv(MultiAgentEnv):
    def __init__(self, num_agents: int = 3):
        self.num_agents = num_agents
        self.static_env_params = CraftaxMARLPixelsEnv.default_static_params()

        self.player_names = [
            f"agent_{i}" for i in range(self.static_env_params.player_count)
        ]

        self.action_spaces = {name: self.action_shape() for name in self.player_names}
        self.observation_spaces = {name: self.observation_shape() for name in self.player_names}

        self.player_names = [
            f"agent_{i}" for i in range(self.static_env_params.player_count)
        ]
        self.player_specific_textures = load_player_specific_textures(
            TEXTURES[BLOCK_PIXEL_SIZE_HUMAN],
            self.static_env_params.player_count
        )

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
        pixels = render_craftax_pixels(
            state, 
            BLOCK_PIXEL_SIZE_HUMAN, 
            self.static_env_params,
            self.player_specific_textures
        ) / 255.0
        return pixels
    
    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
    
    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()
    
    def action_shape(self) -> spaces.Discrete:
        return spaces.Discrete(len(Action) + (self.static_env_params.player_count - 1))
        
    def observation_shape(self) -> spaces.Box:
        map_height = OBS_DIM[0]
        inventory_height = INVENTORY_OBS_HEIGHT
        teammate_dashboard_height = (self.static_env_params.player_count+1)//2
        return spaces.Box(
            0.0,
            1.0,
            (
                OBS_DIM[1] * BLOCK_PIXEL_SIZE_HUMAN,
                (map_height + inventory_height + teammate_dashboard_height) * BLOCK_PIXEL_SIZE_HUMAN,
                3,
            ),
            dtype=jnp.float32,
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
