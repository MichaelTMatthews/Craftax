import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

from gridworld_tech_tree import (
    TECH_TREE_LENGTH,
    OBS_DIM,
    TECH_TREE_OBS_REPEAT,
)
from unet import UNet
from world_model_renderer import WorldModelRenderer
from environment_base.wrappers import FlattenObservationWrapper


def main(path):
    with open(os.path.join(path, "config.yaml")) as f:
        raw_config = yaml.load(f, Loader=yaml.Loader)

        config = {}
        for key, value in raw_config.items():
            if isinstance(value, dict) and "value" in value:
                config[key] = value["value"]

    config["NUM_ENVS"] = 1

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    orbax_checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(max_to_keep=1, create=True)
    checkpoint_manager = CheckpointManager(
        os.path.join(path, "world_models"), orbax_checkpointer, options
    )

    network = UNet()

    if config["ENV_NAME"] == "Gridworld-Tech-Tree-v1":
        from gridworld_tech_tree import Gridworld

        env = Gridworld()
    elif config["ENV_NAME"] == "Gridworld-Random-Maze-v1":
        from gridworld_random_maze import Gridworld

        env = Gridworld()
    elif config["ENV_NAME"] == "Gridworld-Simple-v1":
        from gridworld_simple import Gridworld

        env = Gridworld()
    else:
        raise ValueError(f"Unknown env: {config['ENV_NAME']}")

    env_params = env.default_params
    raw_env = env
    env = FlattenObservationWrapper(env)

    init_x = jnp.zeros((config["NUM_ENVS"], *env.observation_space(env_params).shape))
    init_a = jnp.zeros((config["NUM_ENVS"], env.num_actions))

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng, __rng = jax.random.split(rng, 3)

    def _create_wm(wm_rng):
        wm_params = network.init(wm_rng, init_x, init_a)

        wm_tx = optax.adam(config["LR"])

        wm_train_state = TrainState.create(
            apply_fn=network.apply,
            params=wm_params,
            tx=wm_tx,
        )

        return wm_train_state

    rngs = jax.random.split(rng, config["NUM_WORLD_MODELS"] + 1)
    wm_rngs = rngs[:-1]
    rng = rngs[-1]

    wm_train_states = jax.vmap(_create_wm)(wm_rngs)

    wm_train_states = checkpoint_manager.restore(0, items=wm_train_states)

    train_state1 = jax.tree_map(lambda x: x[0], wm_train_states)
    train_state2 = jax.tree_map(lambda x: x[1], wm_train_states)

    obs, env_state = env.reset(key=__rng)

    gridsquare_render_size = 32

    render_mode = "state"
    gridworld_renderer = WorldModelRenderer(
        raw_env, env_params, gridsquare_render_size, render_mode
    )

    next_env_state = env_state
    collapsed_obs = env.get_obs(env_state)

    obs = jnp.expand_dims(obs, axis=0)
    next_obs = obs
    grad_obs = obs

    latched_action = 0

    while not gridworld_renderer.is_quit_requested():

        # action = jax.random.randint(_rng, shape=(1,), minval=0, maxval=5)
        action = gridworld_renderer.get_action_from_keypress()

        if action == -1:
            action = latched_action
        elif action is not None:
            latched_action = action

            action = jnp.array([action], dtype=jnp.int32)

            action_1h = jax.nn.one_hot(action, num_classes=env.num_actions)
            next_obs_pred1 = network.apply(train_state1.params, obs, action_1h)[0]
            next_obs_pred2 = network.apply(train_state2.params, obs, action_1h)[0]
            collapsed_obs = collapse_obs(next_obs_pred1)

            # Smoothed grad
            smoothing_repeats = 64
            obss = jnp.repeat(obs, smoothing_repeats, axis=0)
            rng, _rng = jax.random.split(rng)
            noisy_obss = obss + jax.random.normal(_rng, obss.shape) * 0.1
            actions_1h = jnp.repeat(action_1h, smoothing_repeats, axis=0)

            grad_obss = calc_grad_obs(
                noisy_obss,
                actions_1h,
                network,
                train_state1.params,
                train_state2.params,
            )

            grad_obs = grad_obss.mean(axis=0)
            grad_obs = jnp.expand_dims(grad_obs, axis=0)

            if jnp.isnan(grad_obs).any():
                print("nan!")

            next_obss = jnp.concatenate(
                (
                    jnp.expand_dims(next_obs_pred1, axis=0),
                    jnp.expand_dims(next_obs_pred2, axis=0),
                ),
                axis=0,
            )
            std = jnp.mean(jnp.std(next_obss, axis=0))
            print(f"std: {std}")

            action = action[0]

            rng, _rng = jax.random.split(rng)
            next_obs, next_env_state, reward, done, info = env.step(
                _rng, env_state, action, env_params
            )
            next_obs = jnp.expand_dims(next_obs, axis=0)

            action = None

        gridworld_renderer.render(
            env, env_state, next_env_state, collapsed_obs, grad_obs
        )

        if action is not None:
            env_state = next_env_state
            obs = next_obs


def calc_grad_obs(obs, action_1h, network, params1, params2):
    # (params, obs, action_1h)

    def _model_disagreement(obs, action_1h, p1, p2):
        next_obs_pred1 = network.apply(p1, obs, action_1h)
        next_obs_pred2 = network.apply(p2, obs, action_1h)
        next_obss = jnp.concatenate(
            (
                jnp.expand_dims(next_obs_pred1, axis=0),
                jnp.expand_dims(next_obs_pred2, axis=0),
            ),
            axis=0,
        )
        std = jnp.mean(jnp.std(next_obss, axis=0))
        return std

    grad_f = jax.grad(_model_disagreement, argnums=0)
    grad_obs = grad_f(obs, action_1h, params1, params2)
    return grad_obs


def collapse_obs(obs):
    grid = obs[: -TECH_TREE_LENGTH * TECH_TREE_OBS_REPEAT]
    tech_tree = obs[-TECH_TREE_LENGTH * TECH_TREE_OBS_REPEAT :]

    grid = jnp.reshape(grid, (OBS_DIM, OBS_DIM, 3 + TECH_TREE_LENGTH))
    grid = jnp.argmax(grid, axis=-1)
    grid = jax.nn.one_hot(grid, num_classes=3 + TECH_TREE_LENGTH)

    tech_tree = (tech_tree > 0.5).astype(int)

    obs = jnp.concatenate((grid.flatten(), tech_tree))
    return obs


if __name__ == "__main__":
    checkpoint = (
        "/home/mikey/PycharmProjects/Craftax/wandb/run-20231121_152444-xksdgdod/files"
    )

    debug = False

    if debug:
        with jax.disable_jit():
            main(checkpoint)
    else:
        main(checkpoint)
