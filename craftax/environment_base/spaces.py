from typing import Any

import jax
import jax.numpy as jnp


# From gymnax
# Credit to the gymnax authors https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/spaces.py


class Space:
    """Minimal jittable class for abstract gymnax space."""

    def sample(self, key: jax.Array) -> jax.Array:
        raise NotImplementedError

    def contains(self, x: jax.Array) -> Any:
        raise NotImplementedError


class Discrete(Space):
    """Minimal jittable class for discrete gymnax spaces."""

    def __init__(self, num_categories: int):
        assert num_categories >= 0
        self.n = num_categories
        self.shape = ()
        self.dtype = jnp.int32

    def sample(self, key: jax.Array) -> jax.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jax.random.randint(
            key, shape=self.shape, minval=0, maxval=self.n
        ).astype(self.dtype)

    def contains(self, x: jax.Array) -> jax.Array:
        """Check whether specific object is within space."""
        x = x.astype(jnp.int32)
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(x >= 0, x < self.n)
        return range_cond


class Box(Space):
    """Minimal jittable class for array-shaped gymnax spaces."""

    def __init__(
        self,
        low: jnp.ndarray | float,
        high: jnp.ndarray | float,
        shape: Any,  # Tuple[int],
        dtype: jnp.dtype = jnp.float32,
    ):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self, key: jax.Array) -> jax.Array:
        """Sample random action uniformly from 1D continuous range."""
        return jax.random.uniform(
            key, shape=self.shape, minval=self.low, maxval=self.high
        ).astype(self.dtype)

    def contains(self, x: jax.Array) -> jax.Array:
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))
        return range_cond
