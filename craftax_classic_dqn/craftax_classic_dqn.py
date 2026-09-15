"""A small, self-contained DQN baseline for Craftax-Classic.

This implementation intentionally uses the symbolic observation and a single
environment.  

Examples:
    python3 craftax_classic_dqn/craftax_classic_dqn.py --smoke-test
    python3 craftax_classic_dqn/craftax_classic_dqn.py --total-steps 100000
    python3 craftax_classic_dqn/craftax_classic_dqn.py --checkpoint dqn.msgpack
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import flax.linen as nn
from flax import serialization
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax

from craftax.craftax_env import make_craftax_env_from_name


class QNetwork(nn.Module):
    """MLP that predicts one Q-value for each discrete action."""

    num_actions: int
    hidden_size: int = 256

    @nn.compact
    def __call__(self, observation: jax.Array) -> jax.Array:
        x = observation.astype(jnp.float32)
        x = nn.relu(nn.Dense(self.hidden_size)(x))
        x = nn.relu(nn.Dense(self.hidden_size)(x))
        return nn.Dense(self.num_actions)(x)


class TransitionBatch(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """Fixed-size replay memory stored in NumPy arrays on the host."""

    def __init__(self, capacity: int, observation_shape: tuple[int, ...]):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive")

        self.capacity = capacity
        self.observations = np.empty(
            (capacity, *observation_shape), dtype=np.float32
        )
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty(capacity, dtype=np.int32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, rng: np.random.Generator, batch_size: int
    ) -> TransitionBatch:
        if self.size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from {self.size} entries"
            )

        indices = rng.integers(0, self.size, size=batch_size)
        return TransitionBatch(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            dones=self.dones[indices],
        )


@dataclass(frozen=True)
class DQNConfig:
    seed: int = 0
    total_steps: int = 100_000
    replay_capacity: int = 50_000
    learning_starts: int = 2_000
    batch_size: int = 64
    train_frequency: int = 4
    target_update_frequency: int = 1_000
    learning_rate: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    hidden_size: int = 256
    max_episode_steps: int = 10_000
    eval_episodes: int = 3
    log_every: int = 1_000
    checkpoint: str | None = None
    smoke_test: bool = False

    def validate(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if self.learning_starts < self.batch_size:
            raise ValueError("learning_starts must be at least batch_size")
        if self.train_frequency <= 0 or self.target_update_frequency <= 0:
            raise ValueError("update frequencies must be positive")
        if self.epsilon_decay_steps <= 0:
            raise ValueError("epsilon_decay_steps must be positive")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between zero and one")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.eval_episodes <= 0:
            raise ValueError("eval_episodes must be positive")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if self.hidden_size <= 0 or self.learning_rate <= 0.0:
            raise ValueError("hidden_size and learning_rate must be positive")


def epsilon_at_step(config: DQNConfig, step: int) -> float:
    fraction = min(step / config.epsilon_decay_steps, 1.0)
    return config.epsilon_start + fraction * (
        config.epsilon_end - config.epsilon_start
    )


@partial(jax.jit, static_argnums=(0,))
def select_greedy_action(
    network: QNetwork, params: Any, observation: jax.Array
) -> jax.Array:
    q_values = network.apply(params, observation[None, ...])[0]
    return jnp.argmax(q_values).astype(jnp.int32)


@jax.jit
def dqn_update(
    state: train_state.TrainState,
    target_params: Any,
    batch: TransitionBatch,
    gamma: float,
) -> tuple[train_state.TrainState, jax.Array, jax.Array, jax.Array]:
    """Perform one vanilla DQN update with a Huber TD loss."""

    next_q_values = state.apply_fn(target_params, batch.next_observations)
    next_q_values = jax.lax.stop_gradient(jnp.max(next_q_values, axis=-1))
    targets = batch.rewards + gamma * (1.0 - batch.dones) * next_q_values

    def loss_fn(params: Any) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        all_q_values = state.apply_fn(params, batch.observations)
        selected_q_values = jnp.take_along_axis(
            all_q_values, batch.actions[:, None], axis=1
        ).squeeze(axis=1)
        td_errors = targets - selected_q_values
        loss = jnp.mean(optax.huber_loss(td_errors, delta=1.0))
        return loss, (jnp.mean(selected_q_values), jnp.mean(jnp.abs(td_errors)))

    (loss, (mean_q, mean_abs_td)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, mean_q, mean_abs_td


def make_env_and_params(max_episode_steps: int):
    # Manual resets are cheaper here than the auto-reset environment, whose
    # functional implementation constructs a candidate reset state every step.
    env = make_craftax_env_from_name(
        "Craftax-Classic-Symbolic-v1", auto_reset=False
    )
    params = env.default_params.replace(max_timesteps=max_episode_steps)
    return env, params


def initialize_agent(
    seed: int,
    observation_shape: tuple[int, ...],
    num_actions: int,
    hidden_size: int,
    learning_rate: float,
) -> tuple[QNetwork, train_state.TrainState, Any]:
    network = QNetwork(num_actions=num_actions, hidden_size=hidden_size)
    dummy_observation = jnp.zeros((1, *observation_shape), dtype=jnp.float32)
    params = network.init(jax.random.PRNGKey(seed), dummy_observation)
    optimizer = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=optimizer,
    )
    return network, state, params


def evaluate(
    network: QNetwork,
    params: Any,
    config: DQNConfig,
) -> dict[str, float]:
    """Evaluate the greedy policy on fresh procedural worlds."""

    env, env_params = make_env_and_params(config.max_episode_steps)
    key = jax.random.PRNGKey(config.seed + 10_000)
    returns: list[float] = []
    lengths: list[int] = []
    scores: list[float] = []

    for _ in range(config.eval_episodes):
        key, reset_key = jax.random.split(key)
        observation, env_state = env.reset(reset_key, env_params)
        episode_return = 0.0

        for episode_step in range(1, config.max_episode_steps + 1):
            action = select_greedy_action(network, params, observation)
            key, step_key = jax.random.split(key)
            observation, env_state, reward, done, info = env.step(
                step_key, env_state, action, env_params
            )
            episode_return += float(reward)

            if bool(done):
                returns.append(episode_return)
                lengths.append(episode_step)
                scores.append(float(info["score"]))
                break
        else:
            raise RuntimeError("Evaluation episode did not terminate at its limit")

    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "length_mean": float(np.mean(lengths)),
        "score_mean": float(np.mean(scores)),
    }


def save_checkpoint(path: str, state: train_state.TrainState) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(serialization.to_bytes(state.params))


def train(config: DQNConfig) -> tuple[train_state.TrainState, dict[str, float]]:
    config.validate()
    env, env_params = make_env_and_params(config.max_episode_steps)
    observation_shape = env.observation_space(env_params).shape
    num_actions = env.num_actions

    network, agent_state, target_params = initialize_agent(
        seed=config.seed,
        observation_shape=observation_shape,
        num_actions=num_actions,
        hidden_size=config.hidden_size,
        learning_rate=config.learning_rate,
    )
    replay = ReplayBuffer(config.replay_capacity, observation_shape)
    numpy_rng = np.random.default_rng(config.seed)

    key = jax.random.PRNGKey(config.seed + 1)
    key, reset_key = jax.random.split(key)
    observation, env_state = env.reset(reset_key, env_params)
    observation_np = np.asarray(observation, dtype=np.float32)

    episode_return = 0.0
    episode_length = 0
    completed_returns: list[float] = []
    latest_loss = math.nan
    latest_mean_q = math.nan
    latest_mean_abs_td = math.nan
    update_count = 0
    started_at = time.perf_counter()

    for step in range(1, config.total_steps + 1):
        epsilon = epsilon_at_step(config, step - 1)
        if numpy_rng.random() < epsilon:
            action = int(numpy_rng.integers(num_actions))
        else:
            action = int(select_greedy_action(network, agent_state.params, observation))

        key, step_key = jax.random.split(key)
        next_observation, next_env_state, reward, done, _ = env.step(
            step_key, env_state, action, env_params
        )
        next_observation_np = np.asarray(next_observation, dtype=np.float32)
        reward_float = float(reward)
        done_bool = bool(done)

        replay.add(
            observation_np,
            action,
            reward_float,
            next_observation_np,
            done_bool,
        )

        episode_return += reward_float
        episode_length += 1

        if (
            step >= config.learning_starts
            and replay.size >= config.batch_size
            and step % config.train_frequency == 0
        ):
            batch = replay.sample(numpy_rng, config.batch_size)
            agent_state, loss, mean_q, mean_abs_td = dqn_update(
                agent_state,
                target_params,
                batch,
                config.gamma,
            )
            latest_loss = float(loss)
            latest_mean_q = float(mean_q)
            latest_mean_abs_td = float(mean_abs_td)
            update_count += 1

        if step % config.target_update_frequency == 0:
            target_params = agent_state.params

        if done_bool:
            completed_returns.append(episode_return)
            print(
                f"episode={len(completed_returns)} step={step} "
                f"length={episode_length} return={episode_return:.3f}"
            )
            key, reset_key = jax.random.split(key)
            observation, env_state = env.reset(reset_key, env_params)
            observation_np = np.asarray(observation, dtype=np.float32)
            episode_return = 0.0
            episode_length = 0
        else:
            observation = next_observation
            env_state = next_env_state
            observation_np = next_observation_np

        if step % config.log_every == 0 or step == config.total_steps:
            elapsed = time.perf_counter() - started_at
            recent_return = (
                float(np.mean(completed_returns[-10:]))
                if completed_returns
                else math.nan
            )
            print(
                f"step={step}/{config.total_steps} epsilon={epsilon:.3f} "
                f"replay={replay.size} updates={update_count} "
                f"loss={latest_loss:.5f} q={latest_mean_q:.3f} "
                f"abs_td={latest_mean_abs_td:.3f} "
                f"return_10={recent_return:.3f} steps_per_second={step / elapsed:.1f}"
            )

    if update_count == 0:
        raise RuntimeError(
            "Training finished without an optimizer update; increase total_steps "
            "or reduce learning_starts"
        )
    if not math.isfinite(latest_loss):
        raise RuntimeError(f"Non-finite final training loss: {latest_loss}")

    evaluation = evaluate(network, agent_state.params, config)
    print(
        "evaluation "
        f"return={evaluation['return_mean']:.3f}±{evaluation['return_std']:.3f} "
        f"length={evaluation['length_mean']:.1f} "
        f"score={evaluation['score_mean']:.3f}"
    )

    if config.checkpoint is not None:
        save_checkpoint(config.checkpoint, agent_state)
        print(f"checkpoint={config.checkpoint}")

    training_metrics = {
        "updates": float(update_count),
        "final_loss": latest_loss,
        "completed_episodes": float(len(completed_returns)),
        **evaluation,
    }
    return agent_state, training_metrics


def smoke_test_config(config: DQNConfig) -> DQNConfig:
    """Reduce the run enough to test the entire pipeline quickly on CPU."""

    return replace(
        config,
        total_steps=64,
        replay_capacity=128,
        learning_starts=8,
        batch_size=8,
        train_frequency=2,
        target_update_frequency=16,
        epsilon_decay_steps=48,
        hidden_size=32,
        max_episode_steps=16,
        eval_episodes=2,
        log_every=16,
        checkpoint=None,
        smoke_test=True,
    )


def parse_args() -> DQNConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--target-update-frequency", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=50_000)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--max-episode-steps", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast end-to-end test instead of a meaningful training run",
    )
    args = parser.parse_args()
    config = DQNConfig(**vars(args))
    return smoke_test_config(config) if args.smoke_test else config


def main() -> None:
    config = parse_args()
    print(f"backend={jax.default_backend()} devices={jax.device_count()}")
    print(config)
    _, metrics = train(config)
    if config.smoke_test:
        assert metrics["updates"] > 0
        assert math.isfinite(metrics["final_loss"])
        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
