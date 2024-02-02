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

from models.actor_critic import ActorCriticConv
from play_craftax_classic import CraftaxRenderer


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
        os.path.join(path, "policies"), orbax_checkpointer, options
    )

    network = ActorCriticConv(17, {})

    if config["ENV_NAME"] == "Craftax-Classic-Symbolic-v1":
        from craftax_classic.envs.craftax_symbolic_env import CraftaxEnv

        env = CraftaxEnv(CraftaxEnv.default_static_params())
    elif config["ENV_NAME"] == "Craftax-Classic-Pixels-v1":
        from craftax_classic.envs.craftax_pixels_env import CraftaxEnv

        env = CraftaxEnv(CraftaxEnv.default_static_params())
    else:
        raise ValueError(f"Unknown env: {config['ENV_NAME']}")

    env_params = env.default_params
    raw_env = env
    # env = FlattenObservationWrapper(env)

    init_x = jnp.zeros((config["NUM_ENVS"], *env.observation_space(env_params).shape))

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng, __rng = jax.random.split(rng, 3)
    network_params = network.init(_rng, init_x)

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

    renderer = CraftaxRenderer(env, env_params)

    while not renderer.is_quit_requested():
        done = np.array([done], dtype=bool)
        # obs = jnp.expand_dims(obs, axis=0)
        obs = jnp.expand_dims(obs, axis=0)

        pi, value = network.apply(train_state.params, obs)
        rng, _rng = jax.random.split(rng)
        action = pi.sample(seed=_rng)[0]
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
        renderer.render(env_state)


if __name__ == "__main__":
    checkpoint = ""

    debug = False

    if debug:
        with jax.disable_jit():
            main(checkpoint)
    else:
        main(checkpoint)
