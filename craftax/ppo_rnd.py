import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from typing import NamedTuple

from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

from craftax.logz.batch_logging import batch_log, create_log_dict
from craftax.models.actor_critic import (
    ActorCritic,
    ActorCriticConv,
)
from craftax.environment_base.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    AutoResetEnvWrapper,
    BatchEnvWrapper,
)
from models.rnd import RNDNetwork, ActorCriticRND


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value_e: jnp.ndarray
    value_i: jnp.ndarray
    reward_e: jnp.ndarray
    reward_i: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    if config["ENV_NAME"] == "Craftax-Classic-Symbolic-v1":
        from craftax.craftax_classic.envs.craftax_symbolic_env import (
            CraftaxClassicSymbolicEnv,
        )

        env = CraftaxClassicSymbolicEnv()
        is_symbolic = True
    elif config["ENV_NAME"] == "Craftax-Classic-Pixels-v1":
        from craftax.craftax_classic.envs.craftax_pixels_env import (
            CraftaxClassicPixelsEnv,
        )

        env = CraftaxClassicPixelsEnv()
        is_symbolic = False
    elif config["ENV_NAME"] == "Craftax-Symbolic-v1":
        from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv

        env = CraftaxSymbolicEnv()
        is_symbolic = True
    elif config["ENV_NAME"] == "Craftax-Pixels-v1":
        from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv

        env = CraftaxPixelsEnv()
        is_symbolic = False
    else:
        raise ValueError(f"Unknown env: {config['ENV_NAME']}")
    env_params = env.default_params

    env = LogWrapper(env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        if is_symbolic:
            network = ActorCriticRND(
                env.action_space(env_params).n, config["LAYER_SIZE"]
            )
        else:
            raise ValueError
            # network = ActorCriticConv(
            #     env.action_space(env_params).n, config["LAYER_SIZE"]
            # )

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
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

        # Exploration state
        ex_state = {
            "rnd_model": None,
        }

        if config["USE_RND"]:
            obs_shape = env.observation_space(env_params).shape
            assert len(obs_shape) == 1, "Only configured for 1D observations"
            obs_shape = obs_shape[0]

            # Random network
            rnd_random_network = RNDNetwork(
                num_layers=3,
                output_dim=config["RND_OUTPUT_SIZE"],
                layer_size=config["RND_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            rnd_random_network_params = rnd_random_network.init(
                _rng, jnp.zeros((1, obs_shape))
            )

            # Distillation Network
            rnd_distillation_network = RNDNetwork(
                num_layers=3,
                output_dim=config["RND_OUTPUT_SIZE"],
                layer_size=config["RND_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            rnd_distillation_network_params = rnd_distillation_network.init(
                _rng, jnp.zeros((1, obs_shape))
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["RND_LR"], eps=1e-5),
            )
            ex_state["rnd_distillation_network"] = TrainState.create(
                apply_fn=rnd_distillation_network.apply,
                params=rnd_distillation_network_params,
                tx=tx,
            )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    ex_state,
                    rng,
                    update_step,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value_e, value_i = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward_e, done, info = env.step(
                    _rng, env_state, action, env_params
                )

                reward_i = jnp.zeros(config["NUM_ENVS"])

                if config["USE_RND"]:
                    random_pred = rnd_random_network.apply(
                        rnd_random_network_params, obsv
                    )

                    distill_pred = ex_state["rnd_distillation_network"].apply_fn(
                        ex_state["rnd_distillation_network"].params, obsv
                    )
                    error = (random_pred - distill_pred) * (1 - done[:, None])
                    mse = jnp.square(error).mean(axis=-1)

                    reward_i = mse * config["RND_REWARD_COEFF"]

                reward = reward_e + reward_i

                transition = Transition(
                    done=done,
                    action=action,
                    value_e=value_e,
                    value_i=value_i,
                    reward=reward,
                    reward_i=reward_i,
                    reward_e=reward_e,
                    log_prob=log_prob,
                    obs=last_obs,
                    next_obs=obsv,
                    info=info,
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    ex_state,
                    rng,
                    update_step,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step,
            ) = runner_state
            _, last_val_e, last_val_i = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val, is_extrinsic):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value, is_extrinsic = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        jax.lax.select(
                            is_extrinsic, transition.value_e, transition.value_i
                        ),
                        transition.reward,
                    )
                    done = jnp.logical_and(
                        done, jnp.logical_or(config["RND_IS_EPISODIC"], is_extrinsic)
                    )

                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value, is_extrinsic), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, is_extrinsic),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + jax.lax.select(
                    is_extrinsic, traj_batch.value_e, traj_batch.value_i
                )

            advantages_e, targets_e = _calculate_gae(traj_batch, last_val_e, True)
            advantages_i, targets_i = _calculate_gae(traj_batch, last_val_e, False)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    (
                        traj_batch,
                        advantages_e,
                        targets_e,
                        advantages_i,
                        targets_i,
                    ) = batch_info

                    # Policy/value network
                    def _loss_fn(
                        params, traj_batch, gae_e, targets_e, gae_i, targets_i
                    ):
                        # RERUN NETWORK
                        pi, value_e, value_i = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE EXTRINSIC VALUE LOSS
                        value_pred_clipped_e = traj_batch.value_e + (
                            value_e - traj_batch.value_e
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_e = jnp.square(value_e - targets_e)
                        value_losses_clipped_e = jnp.square(
                            value_pred_clipped_e - targets_e
                        )
                        value_loss_e = (
                            0.5
                            * jnp.maximum(value_losses_e, value_losses_clipped_e).mean()
                        )

                        # CALCULATE INTRINSIC VALUE LOSS
                        value_pred_clipped_i = traj_batch.value_i + (
                            value_i - traj_batch.value_i
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_i = jnp.square(value_i - targets_i)
                        value_losses_clipped_i = jnp.square(
                            value_pred_clipped_i - targets_i
                        )
                        value_loss_i = (
                            0.5
                            * jnp.maximum(value_losses_i, value_losses_clipped_i).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        gae = gae_e + gae_i
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * (value_loss_e + value_loss_i)
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (
                            value_loss_e,
                            value_loss_i,
                            loss_actor,
                            entropy,
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params,
                        traj_batch,
                        advantages_e,
                        targets_e,
                        advantages_i,
                        targets_i,
                    )
                    train_state = train_state.apply_gradients(grads=grads)

                    losses = (total_loss, 0)
                    return train_state, losses

                (
                    train_state,
                    traj_batch,
                    advantages_e,
                    targets_e,
                    advantages_i,
                    targets_i,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (
                    traj_batch,
                    advantages_e,
                    targets_e,
                    advantages_i,
                    targets_i,
                )
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, losses = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    traj_batch,
                    advantages_e,
                    targets_e,
                    advantages_i,
                    targets_i,
                    rng,
                )
                return update_state, losses

            update_state = (
                train_state,
                traj_batch,
                advantages_e,
                targets_e,
                advantages_i,
                targets_i,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]
            metric = jax.tree_map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )

            rng = update_state[-1]

            # UPDATE EXPLORATION STATE
            def _update_ex_epoch(update_state, unused):
                def _update_ex_minbatch(ex_state, traj_batch):
                    rnd_loss = 0

                    if config["USE_RND"]:

                        def _rnd_loss_fn(rnd_distillation_params, traj_batch):
                            random_network_out = rnd_random_network.apply(
                                rnd_random_network_params, traj_batch.next_obs
                            )

                            distillation_network_out = ex_state[
                                "rnd_distillation_network"
                            ].apply_fn(rnd_distillation_params, traj_batch.next_obs)

                            error = (random_network_out - distillation_network_out) * (
                                1 - traj_batch.done[:, None]
                            )
                            return jnp.square(error).mean() * config["RND_LOSS_COEFF"]

                        rnd_grad_fn = jax.value_and_grad(_rnd_loss_fn, has_aux=False)
                        rnd_loss, rnd_grad = rnd_grad_fn(
                            ex_state["rnd_distillation_network"].params, traj_batch
                        )
                        ex_state["rnd_distillation_network"] = ex_state[
                            "rnd_distillation_network"
                        ].apply_gradients(grads=rnd_grad)

                    losses = (rnd_loss,)
                    return ex_state, losses

                (ex_state, traj_batch, rng) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                ex_state, losses = jax.lax.scan(
                    _update_ex_minbatch, ex_state, minibatches
                )
                update_state = (ex_state, traj_batch, rng)
                return update_state, losses

            if config["USE_RND"]:
                ex_update_state = (ex_state, traj_batch, rng)
                ex_update_state, ex_loss = jax.lax.scan(
                    _update_ex_epoch,
                    ex_update_state,
                    None,
                    config["EXPLORATION_UPDATE_EPOCHS"],
                )
                metric["rnd_loss"] = ex_loss[0].mean()
                metric["reward_i"] = traj_batch.reward_i.mean()
                metric["reward_e"] = traj_batch.reward_e.mean()

                ex_state = ex_update_state[0]
                rng = ex_update_state[-1]

            # wandb logging
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    batch_log(update_step, to_log, config)

                jax.debug.callback(
                    callback,
                    metric,
                    update_step,
                )

            runner_state = (
                train_state,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step + 1,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            ex_state,
            _rng,
            0,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}  # , "info": metric}

    return train


def run_ppo(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"]
            + "-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M",
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_jit = jax.jit(make_train(config))
    train_vmap = jax.vmap(train_jit)

    t0 = time.time()
    out = train_vmap(rngs)
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))
    # t1 = time.time()
    # out = train_vmap(rngs)
    # t2 = time.time()
    # print("t2", t2 - t1)
    # print("SPS2: ", config["TOTAL_TIMESTEPS"] / (t2 - t1))

    if config["USE_WANDB"]:
        # if config["DEBUG"] == "end":
        #     info = out["info"]
        #     for update in range(info["timestep"].shape[1]):
        #         if update % 10 == 0:
        #             for repeat in range(info["timestep"].shape[0]):
        #                 update_info = jax.tree_map(lambda x: x[repeat, update], info)
        #                 to_log = create_log_dict(update_info)
        #                 batch_log(update, to_log, config)
        #
        #     t2 = time.time()
        #     print("Time to log to wandb", t2 - t1)

        def _save_network(rs_index, dir_name):
            train_states = out["runner_state"][rs_index]
            train_state = jax.tree_map(lambda x: x[0], train_states)
            orbax_checkpointer = PyTreeCheckpointer()
            options = CheckpointManagerOptions(max_to_keep=1, create=True)
            path = os.path.join(wandb.run.dir, dir_name)
            checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
            print(f"saved runner state to {path}")
            save_args = orbax_utils.save_args_from_target(train_state)
            checkpoint_manager.save(
                config["TOTAL_TIMESTEPS"],
                train_state,
                save_kwargs={"save_args": save_args},
            )

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--total_timesteps", type=lambda x: int(float(x)), default=1e9
    )  # Allow scientific notation
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--save_policy", action="store_true")
    parser.add_argument("--num_repeats", type=int, default=8)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

    # EXPLORATION
    parser.add_argument("--exploration_update_epochs", type=int, default=1)
    # RND
    parser.add_argument(
        "--use_rnd", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--rnd_layer_size", type=int, default=256)
    parser.add_argument("--rnd_output_size", type=int, default=512)
    parser.add_argument("--rnd_lr", type=float, default=3e-4)
    parser.add_argument("--rnd_reward_coeff", type=float, default=0.1)
    parser.add_argument("--rnd_loss_coeff", type=float, default=0.001)
    parser.add_argument(
        "--rnd_is_episodic", action=argparse.BooleanOptionalAction, default=False
    )

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)
