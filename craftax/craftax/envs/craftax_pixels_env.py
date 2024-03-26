from jax import lax
from gymnax.environments import spaces
from typing import Tuple, Optional
import chex

from craftax.craftax.envs.common import compute_score
from craftax.craftax.constants import *
from craftax.craftax.game_logic import craftax_step
from craftax.craftax.craftax_state import EnvState, EnvParams, StaticEnvParams
from craftax.craftax.renderer import render_craftax_pixels
from craftax.craftax.util.game_logic_utils import has_beaten_boss
from craftax.craftax.world_gen.world_gen import generate_world
from craftax.environment_base.environment_no_auto_reset import EnvironmentNoAutoReset


class CraftaxPixelsEnv(EnvironmentNoAutoReset):
    def __init__(self, static_env_params: Optional[StaticEnvParams] = None):
        super().__init__()

        if static_env_params is None:
            static_env_params = CraftaxPixelsEnv.default_static_params()
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
        done_steps = state.timestep >= params.max_timesteps
        is_dead = state.player_health <= 0
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
                OBS_DIM[1] * BLOCK_PIXEL_SIZE_AGENT,
                (OBS_DIM[0] + INVENTORY_OBS_HEIGHT) * BLOCK_PIXEL_SIZE_AGENT,
                3,
            ),
            dtype=jnp.int32,
        )
