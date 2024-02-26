import jax.numpy as jnp
import flax.linen as nn


class FCNetwork(nn.Module):
    layer_size: int
    output_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        activation = nn.relu

        # TODO Look at weight inits

        emb = x
        for _ in range(self.num_layers):
            emb = nn.Dense(
                self.layer_size,
            )(emb)
            emb = activation(emb)

        emb = nn.Dense(self.output_dim)(emb)

        return emb
