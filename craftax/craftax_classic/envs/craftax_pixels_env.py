from jax import lax
from gymnax.environments import spaces, environment
from typing import Tuple, Optional
import chex

from craftax.environment_base.environment_bases import EnvironmentNoAutoReset
from craftax.craftax_classic.envs.common import compute_score
from craftax.craftax_classic.constants import *
from craftax.craftax_classic.game_logic import craftax_step, is_game_over
from craftax.craftax_classic.envs.craftax_state import (
    EnvState,
    EnvParams,
    StaticEnvParams,
)
from craftax.craftax_classic.renderer import render_craftax_pixels
from craftax.craftax_classic.world_gen import generate_world


class CraftaxClassicPixelsEnvNoAutoReset(EnvironmentNoAutoReset):
    def __init__(self, static_env_params: StaticEnvParams = None):
        super().__init__()

        if static_env_params is None:
            static_env_params = self.default_static_params()
        self.static_env_params = static_env_params

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:

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
        pixels = render_craftax_pixels(state, BLOCK_PIXEL_SIZE_AGENT) / 255.0
        return pixels

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return is_game_over(state, params)

    @property
    def name(self) -> str:
        return "Craftax-Classic-Pixels-NoAutoReset-v1"

    @property
    def num_actions(self) -> int:
        return 17

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(17)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(
            0.0,
            1.0,
            (
                OBS_DIM[1] * BLOCK_PIXEL_SIZE_AGENT,
                (OBS_DIM[0] + INVENTORY_OBS_HEIGHT) * BLOCK_PIXEL_SIZE_AGENT,
                3,
            ),
            dtype=jnp.float32,
        )


class CraftaxClassicPixelsEnv(environment.Environment):
    def __init__(self, static_env_params: StaticEnvParams = None):
        super().__init__()

        if static_env_params is None:
            static_env_params = self.default_static_params()
        self.static_env_params = static_env_params

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:

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
        pixels = render_craftax_pixels(state, BLOCK_PIXEL_SIZE_AGENT) / 255.0
        return pixels

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return is_game_over(state, params)

    @property
    def name(self) -> str:
        return "Craftax-Classic-Pixels-v1"

    @property
    def num_actions(self) -> int:
        return 17

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(17)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(
            0.0,
            1.0,
            (
                OBS_DIM[1] * BLOCK_PIXEL_SIZE_AGENT,
                (OBS_DIM[0] + INVENTORY_OBS_HEIGHT) * BLOCK_PIXEL_SIZE_AGENT,
                3,
            ),
            dtype=jnp.float32,
        )
