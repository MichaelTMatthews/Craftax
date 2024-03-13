import jax
import jax.numpy as jnp

# For utility functions - functions called more than once in meaningfully different parts of the codebase
# With the additional constraint that the functions make no reference (i.e. don't import from) any Craftax code


def get_distance_map(position, map_size):
    dist_x = jnp.abs(jnp.arange(0, map_size[0]) - position[0])
    dist_x = jnp.expand_dims(dist_x, axis=1)
    dist_x = jnp.tile(dist_x, (1, map_size[1]))

    dist_y = jnp.abs(jnp.arange(0, map_size[1]) - position[1])
    dist_y = jnp.expand_dims(dist_y, axis=0)
    dist_y = jnp.tile(dist_y, (map_size[0], 1))

    coords = jnp.stack([dist_x, dist_y], axis=-1)

    def _euclid_distance(x):
        return jnp.sqrt(x[0] ** 2 + x[1] ** 2)

    dist = jax.vmap(jax.vmap(_euclid_distance))(coords)

    return dist
