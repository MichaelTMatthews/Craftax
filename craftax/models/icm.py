import jax
import jax.numpy as jnp
import flax.linen as nn


class ICMEncoder(nn.Module):
    layer_size: int
    output_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, obs):
        activation = nn.relu

        # TODO Look at weight inits

        emb = obs
        for _ in range(self.num_layers):
            emb = nn.Dense(
                self.layer_size,
            )(emb)
            emb = activation(emb)

        emb = nn.Dense(self.output_dim)(emb)

        return emb


class ICMForward(nn.Module):
    layer_size: int
    output_dim: int
    num_layers: int
    num_actions: int

    @nn.compact
    def __call__(self, latent, action):
        activation = nn.relu

        action1h = jax.nn.one_hot(action, num_classes=self.num_actions)
        emb = jnp.concatenate((latent, action1h), axis=-1)
        for _ in range(self.num_layers):
            emb = nn.Dense(
                self.layer_size,
            )(emb)
            emb = activation(emb)

        emb = nn.Dense(self.output_dim)(emb)

        return emb


class ICMInverse(nn.Module):
    layer_size: int
    output_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, latent, next_latent):
        activation = nn.relu

        emb = jnp.concatenate((latent, next_latent), axis=-1)
        for _ in range(self.num_layers):
            emb = nn.Dense(
                self.layer_size,
            )(emb)
            emb = activation(emb)

        action_raw = nn.Dense(self.output_dim)(emb)

        action_logits = jax.nn.log_softmax(action_raw)

        return action_logits
