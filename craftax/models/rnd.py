import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence

import distrax


class RNDNetwork(nn.Module):
    layer_size: int
    output_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        activation = nn.relu

        emb = x
        for _ in range(self.num_layers):
            emb = nn.Dense(
                self.layer_size,
            )(emb)
            emb = activation(emb)

        emb = nn.Dense(self.output_dim)(emb)

        return emb


class ActorCriticRND(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Extrinsic reward
        critic_e = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic_e = activation(critic_e)

        critic_e = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic_e)
        critic_e = activation(critic_e)

        critic_e = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic_e)
        critic_e = activation(critic_e)

        critic_e = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic_e
        )

        # Intrinsic reward
        critic_i = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic_i = activation(critic_i)

        critic_i = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic_i)
        critic_i = activation(critic_i)

        critic_i = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic_i)
        critic_i = activation(critic_i)

        critic_i = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic_i
        )

        return pi, jnp.squeeze(critic_e, axis=-1), jnp.squeeze(critic_i, axis=-1)
