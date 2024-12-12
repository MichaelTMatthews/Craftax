from jax import lax
from gymnax.environments import spaces, environment
from typing import Tuple, Optional
import chex

from craftax_marl.envs.common import compute_score
from craftax_marl.constants import *
from craftax_marl.game_logic import craftax_step
from craftax_marl.craftax_state import EnvState, EnvParams, StaticEnvParams
from craftax_marl.renderer import render_craftax_pixels
from craftax_marl.util.game_logic_utils import has_beaten_boss
from craftax_marl.world_gen.world_gen import generate_world
from environment_base.environment_bases import EnvironmentNoAutoReset


class CraftaxMARLPixelsEnvNoAutoReset(EnvironmentNoAutoReset):
    def __init__(self, static_env_params: Optional[StaticEnvParams] = None):
        super().__init__()

        if static_env_params is None:
            static_env_params = CraftaxMARLPixelsEnvNoAutoReset.default_static_params()
        self.static_env_params = static_env_params
        self.player_names = [
            f"agent_{i}" for i in range(self.static_env_params.player_count)
        ]
        self.player_specific_textures = load_player_specific_textures(
            TEXTURES[BLOCK_PIXEL_SIZE_HUMAN],
            self.static_env_params.player_count
        )

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:

        actions = jnp.array(list(actions.values()))
        state, reward = craftax_step(key, state, action, params, self.static_env_params)

        done = self.is_terminal(state, params)
        info = compute_score(state, done)
        info["discount"] = self.discount(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, rng: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        state = generate_world(rng, params, self.static_env_params)

        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        pixels = render_craftax_pixels(
            state, 
            BLOCK_PIXEL_SIZE_HUMAN, 
            self.static_env_params,
            self.player_specific_textures,
        ) / 255.0
        obs = {player: pixels[i] for i, player in enumerate(self.player_names)}
        return obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        done_steps = state.timestep >= params.max_timesteps
        is_dead = jnp.logical_not(state.player_alive).all()
        defeated_boss = has_beaten_boss(state, self.static_env_params)
        is_terminal = jnp.logical_or(is_dead, done_steps)
        is_terminal = jnp.logical_or(is_terminal, defeated_boss)
        return is_terminal

    @property
    def name(self) -> str:
        return "Craftax-Pixels-v1"

    @property
    def num_actions(self) -> int:
        return len(Action)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(len(Action))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(
            0.0,
            1.0,
            (
                OBS_DIM[1] * BLOCK_PIXEL_SIZE_HUMAN,
                (OBS_DIM[0] + INVENTORY_OBS_HEIGHT) * BLOCK_PIXEL_SIZE_HUMAN,
                3,
            ),
            dtype=jnp.int32,
        )


class CraftaxMARLPixelsEnv(environment.Environment):
    def __init__(self, static_env_params: Optional[StaticEnvParams] = None):
        super().__init__()

        if static_env_params is None:
            static_env_params = CraftaxMARLPixelsEnv.default_static_params()
        self.static_env_params = static_env_params
        self.player_names = [
            f"agent_{i}" for i in range(self.static_env_params.player_count)
        ]
        self.player_specific_textures = load_player_specific_textures(
            TEXTURES[BLOCK_PIXEL_SIZE_HUMAN],
            self.static_env_params.player_count
        )

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:

        actions = jnp.array(list(actions.values()))
        state, reward = craftax_step(key, state, action, params, self.static_env_params)

        done = self.is_terminal(state, params)
        info = compute_score(state, done)
        info["discount"] = self.discount(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, rng: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        state = generate_world(rng, params, self.static_env_params)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        pixels = render_craftax_pixels(
            state, 
            BLOCK_PIXEL_SIZE_HUMAN, 
            self.static_env_params,
            self.player_specific_textures
        ) / 255.0
        obs = {player: pixels[i] for i, player in enumerate(self.player_names)}
        return obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        done_steps = state.timestep >= params.max_timesteps
        is_dead = jnp.logical_not(state.player_alive).all()
        defeated_boss = has_beaten_boss(state, self.static_env_params)
        is_terminal = jnp.logical_or(is_dead, done_steps)
        is_terminal = jnp.logical_or(is_terminal, defeated_boss)
        return is_terminal

    @property
    def name(self) -> str:
        return "Craftax-MARL-Pixels-v1"

    @property
    def num_actions(self) -> int:
        return len(Action)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(len(Action))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(
            0.0,
            1.0,
            (
                OBS_DIM[1] * BLOCK_PIXEL_SIZE_HUMAN,
                (OBS_DIM[0] + INVENTORY_OBS_HEIGHT) * BLOCK_PIXEL_SIZE_HUMAN,
                3,
            ),
            dtype=jnp.int32,
        )
