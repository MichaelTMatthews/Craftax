import jax
from jax import lax
from gymnax.environments import spaces
from typing import Tuple, Optional
import chex

from .common import compute_score
from .environment import EnvironmentNoAutoReset
from ..constants import *
from ..game_logic import craftax_step
from ..envs import Environment
from ..craftax_state import EnvState, EnvParams, StaticEnvParams
from ..renderer import render_craftax_symbolic
from ..util.game_logic_utils import has_beaten_boss
from ..world_gen.world_gen import generate_world


class CraftaxEnv(EnvironmentNoAutoReset):
    def __init__(self, static_env_params: Optional[StaticEnvParams] = None):
        super().__init__()

        if static_env_params is None:
            static_env_params = CraftaxEnv.default_static_params()
        self.static_env_params = static_env_params

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    def step_env(
        self, rng: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        state, reward = craftax_step(rng, state, action, params, self.static_env_params)

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
        rng, _rng = jax.random.split(rng)
        state = generate_world(_rng, params, self.static_env_params)

        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        pixels = render_craftax_symbolic(state)
        return pixels

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        done_steps = state.timestep >= params.max_timesteps
        in_lava = (
            state.map[
                state.player_level,
                state.player_position[0],
                state.player_position[1],
            ]
            == BlockType.LAVA.value
        )
        is_dead = state.player_health <= 0
        defeated_boss = has_beaten_boss(state, self.static_env_params)

        is_terminal = jnp.logical_or(done_steps, in_lava)
        is_terminal = jnp.logical_or(is_terminal, is_dead)
        is_terminal = jnp.logical_or(is_terminal, defeated_boss)

        return is_terminal

    @property
    def name(self) -> str:
        return "Craftax-Symbolic-v1"

    @property
    def num_actions(self) -> int:
        return len(Action)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(len(Action))

    @staticmethod
    def get_map_obs_shape():
        num_mob_classes = 5
        num_mob_types = 8
        num_blocks = len(BlockType)

        return OBS_DIM[0], OBS_DIM[1], num_blocks + num_mob_classes * num_mob_types + 1

    @staticmethod
    def get_flat_map_obs_shape():
        map_obs_shape = CraftaxEnv.get_map_obs_shape()
        return map_obs_shape[0] * map_obs_shape[1] * map_obs_shape[2]

    @staticmethod
    def get_inventory_obs_shape():
        return 50

    def observation_space(self, params: EnvParams) -> spaces.Box:
        flat_map_obs_shape = self.get_flat_map_obs_shape()
        inventory_obs_shape = self.get_inventory_obs_shape()

        obs_shape = flat_map_obs_shape + inventory_obs_shape
        # obs_shape = 8168

        return spaces.Box(
            0.0,
            1.0,
            (obs_shape,),
            dtype=jnp.int32,
        )
