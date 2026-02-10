from dataclasses import dataclass

import jax
from typing import Union
from functools import partial


class EnvironmentNoAutoReset(object):
    """Similar to the base Gymnax environment but without auto-resets."""

    @property
    def default_params(self):
        return NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: jax.Array,
        state,
        action: Union[int, float],
        params=None,
    ):
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        obs, state, reward, done, info = self.step_env(key, state, action, params)
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.Array, params=None):
        """Performs resetting of environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        obs, state = self.reset_env(key, params)
        return obs, state

    def step_env(
        self,
        key: jax.Array,
        state,
        action: Union[int, float],
        params,
    ):
        """Environment-specific step transition."""
        raise NotImplementedError

    def reset_env(self, key: jax.Array, params):
        """Environment-specific reset."""
        raise NotImplementedError

    def get_obs(self, state) -> jax.Array:
        """Applies observation function to state."""
        raise NotImplementedError

    def is_terminal(self, state, params) -> bool:
        """Check whether state transition is terminal."""
        raise NotImplementedError

    def discount(self, state, params) -> float:
        """Return a discount of zero if the episode has terminated."""
        return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        raise NotImplementedError

    def action_space(self, params):
        """Action space of the environment."""
        raise NotImplementedError

    def observation_space(self, params):
        """Observation space of the environment."""
        raise NotImplementedError

    def state_space(self, params):
        """State space of the environment."""
        raise NotImplementedError
