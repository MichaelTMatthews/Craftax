import jax
import jax.numpy as jnp
from flax import linen as nn

from tech_tree_gridworld.gridworld_tech_tree import (
    Gridworld,
    TECH_TREE_LENGTH,
    OBS_DIM,
    TECH_TREE_OBS_REPEAT,
)


class Encoder(nn.Module):
    features: int
    n_blocks: int
    n_groups: int
    mlp_features: int = 8

    @nn.compact
    def __call__(self, x, t):
        zs = []
        t = t * nn.sigmoid(t)
        for i in range(self.n_blocks):
            x = nn.Conv(self.features * (2**i), kernel_size=(3, 3))(x)

            t_emb = nn.Dense(self.mlp_features)(t)
            t_emb = jnp.expand_dims(t_emb, 1)
            t_emb = jnp.expand_dims(t_emb, 1)
            t_emb = jnp.repeat(t_emb, x.shape[1], axis=1)
            t_emb = jnp.repeat(t_emb, x.shape[2], axis=2)
            x = jnp.concatenate((x, t_emb), axis=-1)

            x = nn.GroupNorm(num_groups=self.n_groups)(x)
            x = nn.relu(x)
            x = nn.Conv(self.features * (2**i), kernel_size=(3, 3))(x)
            x = nn.GroupNorm(num_groups=self.n_groups)(x)
            x = nn.relu(x)
            zs.append(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return zs


class Decoder(nn.Module):
    out_features: int
    features: int
    n_blocks: int
    n_groups: int
    mlp_features: int = 8

    def _upsample(self, x, target_length):
        # Deconvolution currently just duplicates elements
        # TODO: Test alternative upsampling methods
        return jax.image.resize(
            x, shape=(*x.shape[:-2], target_length, x.shape[-1]), method="nearest"
        )

    @nn.compact
    def __call__(self, zs, t):
        x = zs[-1]
        t = t * nn.sigmoid(t)
        for i in range(self.n_blocks - 2, -1, -1):
            z = zs[i]

            # x = self._upsample(x, z.shape[-2])

            x = jax.image.resize(
                x,
                shape=(*x.shape[:-3], z.shape[-3], z.shape[-2], x.shape[-1]),
                method="nearest",
            )
            x = nn.Conv(self.features * (2**i), kernel_size=(2, 2))(x)

            t_emb = nn.Dense(self.mlp_features)(t)
            t_emb = jnp.expand_dims(t_emb, 1)
            t_emb = jnp.expand_dims(t_emb, 1)
            t_emb = jnp.repeat(t_emb, x.shape[1], axis=1)
            t_emb = jnp.repeat(t_emb, x.shape[2], axis=2)
            x = jnp.concatenate((x, t_emb), axis=-1)

            x = nn.GroupNorm(num_groups=self.n_groups)(x)
            x = nn.relu(x)
            x = jnp.concatenate([x, z], axis=-1)
            x = nn.Conv(self.features * (2**i), kernel_size=(3, 3))(x)
            x = nn.GroupNorm(num_groups=self.n_groups)(x)
            x = nn.relu(x)
            x = nn.Conv(self.features * (2**i), kernel_size=(3, 3))(x)
            x = nn.GroupNorm(num_groups=self.n_groups)(x)
            x = nn.relu(x)
        x = nn.Conv(self.out_features, kernel_size=(1, 1))(x)
        return x


class UNet(nn.Module):
    features: int = 16
    n_blocks: int = 4
    n_groups: int = 1

    @nn.compact
    def __call__(self, obs, action):
        obs_for_mlp = obs[:, -TECH_TREE_LENGTH * TECH_TREE_OBS_REPEAT :]
        mlp_in = jnp.concatenate((obs_for_mlp, action), axis=-1)

        obs_for_conv = obs[:, : -TECH_TREE_LENGTH * TECH_TREE_OBS_REPEAT]
        obs_for_conv = obs_for_conv.reshape(
            (
                obs_for_mlp.shape[0],
                OBS_DIM,
                OBS_DIM,
                TECH_TREE_LENGTH + 3,
            )
        )

        zs = Encoder(
            features=self.features, n_blocks=self.n_blocks, n_groups=self.n_groups
        )(obs_for_conv, mlp_in)
        y = Decoder(
            out_features=obs_for_conv.shape[-1],
            features=self.features,
            n_blocks=self.n_blocks,
            n_groups=self.n_groups,
        )(zs, mlp_in)

        flattened_conv = jnp.reshape(
            y, obs[:, : -TECH_TREE_LENGTH * TECH_TREE_OBS_REPEAT].shape
        )

        conv1 = jnp.reshape(zs[-1], (zs[-1].shape[0], -1))
        conv2 = jnp.reshape(zs[0], (zs[0].shape[0], -1))
        tt_emb = jnp.concatenate((conv1, conv2, mlp_in), axis=-1)

        tt_emb = nn.Dense(256)(tt_emb)
        tt_emb = nn.relu(tt_emb)

        tt_emb = nn.Dense(128)(tt_emb)
        tt_emb = nn.relu(tt_emb)

        tt_emb = nn.Dense(64)(tt_emb)
        tt_emb = nn.relu(tt_emb)

        tt_emb = nn.Dense(TECH_TREE_LENGTH * TECH_TREE_OBS_REPEAT)(tt_emb)
        tt_emb = nn.sigmoid(tt_emb)

        return jnp.concatenate((flattened_conv, tt_emb), axis=-1)


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    env = Gridworld()
    env_params = env.default_params

    network = UNet()
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((1, env.observation_space(env_params).shape))
    init_a = jnp.zeros((1, env.num_actions))
    wm_params = network.init(_rng, init_x, init_a)
