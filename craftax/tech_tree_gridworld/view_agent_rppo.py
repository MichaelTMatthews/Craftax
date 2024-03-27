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

from gridworld_renderer import GridworldRenderer
from gridworld_tech_tree import MAP_SIZE, TECH_TREE_LENGTH, EnvState
from environment_base.wrappers import FlattenObservationWrapper


def main(path):
    ckpt_path = os.path.join(path, "checkpoints/0/default")

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
        os.path.join(path, "checkpoints"), orbax_checkpointer, options
    )

    network = ActorCriticConvRNN(5, {})

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

    init_x = (
        jnp.zeros((1, config["NUM_ENVS"], *env.observation_space(env_params).shape)),
        jnp.zeros((1, config["NUM_ENVS"])),
    )
    init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], LAYER_WIDTH)
    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng, __rng = jax.random.split(rng, 3)
    network_params = network.init(_rng, init_hstate, init_x)

    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    # orbax_checkpointer.restore(ckpt_path, item=train_state)
    train_state = checkpoint_manager.restore(0, items=train_state)

    obs, env_state = env.reset(key=__rng)
    done = 0

    #
    bellwether_env = False
    if bellwether_env:
        position = jnp.array([0, 7], dtype=jnp.int32)

        grid = jnp.zeros(MAP_SIZE, dtype=jnp.int32)
        for i in range(TECH_TREE_LENGTH):
            grid = grid.at[i + 1, 7].set(i + 2)

        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if np.random.random() < 0.2 and grid[x, y] == 0:
                    grid = grid.at[x, y].set(1)

        completed_techs = env_state.completed_techs

        env_state = EnvState(grid, position, completed_techs, 0)

    #

    hstate = init_hstate

    gridsquare_render_size = 32

    render_mode = "state"
    gridworld_renderer = GridworldRenderer(
        raw_env, env_params, gridsquare_render_size, render_mode
    )

    while not gridworld_renderer.is_quit_requested():
        done = np.array([[done]], dtype=bool)
        obs = jnp.expand_dims(obs, axis=0)
        obs = jnp.expand_dims(obs, axis=0)

        hstate, pi, value = network.apply(train_state.params, hstate, (obs, done))
        rng, _rng = jax.random.split(rng)
        action = pi.sample(seed=_rng)[0, 0]
        # action = jnp.argmax(pi.probs[0, 0])

        if action is not None:
            rng, _rng = jax.random.split(rng)
            obs, env_state, reward, done, info = env.step(
                _rng, env_state, action, env_params
            )
            # print(obs.T)
            if reward > 0:
                print(reward)
            if done:
                print("\n")
        gridworld_renderer.render(env_state)


if __name__ == "__main__":
    checkpoint = (
        "/home/mikey/PycharmProjects/Craftax/wandb/run-20231102_141239-zs1abfdc/files"
    )

    debug = True

    if debug:
        with jax.disable_jit():
            main(checkpoint)
    else:
        main(checkpoint)
