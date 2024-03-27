import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional, Sequence
import chex
from flax import struct

OBS_DIM = 9
assert OBS_DIM % 2 == 1
MAP_SIZE = (5, 5)
TECH_TREE_LENGTH = 5
GRID_WORLD_DIRECTIONS = jnp.array(
    [[0, -1], [1, 0], [0, 1], [-1, 0]] + [[0, 0] for _ in range(TECH_TREE_LENGTH)]
)
REWARD_EVERY_N_TECHS = 5

FULLY_OBSERVABLE = False
TECH_TREE_OBS_REPEAT = 30


@struct.dataclass
class EnvState:
    grid: jnp.ndarray
    position: jnp.ndarray
    completed_techs: jnp.ndarray
    timestep: int


@struct.dataclass
class EnvParams:
    max_timesteps: int = 512
    wall_gen_threshold: float = 0.9


class Gridworld(environment.Environment):
    def __init__(self):
        super().__init__()
        if FULLY_OBSERVABLE:
            self.obs_shape = MAP_SIZE
        else:
            self.obs_shape = (OBS_DIM, OBS_DIM)

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for CartPole-v1
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""

        proposed_position = state.position + GRID_WORLD_DIRECTIONS[action]

        in_bounds_x = jnp.logical_and(
            0 <= proposed_position[0], proposed_position[0] < state.grid.shape[0]
        )
        in_bounds_y = jnp.logical_and(
            0 <= proposed_position[1], proposed_position[1] < state.grid.shape[1]
        )
        in_bounds = jnp.logical_and(in_bounds_x, in_bounds_y)

        # in_wall = state.grid[proposed_position[0], proposed_position[1]] == 1
        in_wall = False
        valid_move = jnp.logical_and(in_bounds, jnp.logical_not(in_wall))

        position = (
            state.position + valid_move.astype(int) * GRID_WORLD_DIRECTIONS[action]
        )

        # reward = -0.01
        reward = 0.0

        # Tech tree reward TODO
        tech_on = state.grid[position[0], position[1]] - 2
        techs_on_this_timestep = jnp.zeros(TECH_TREE_LENGTH, dtype=jnp.int32)
        techs_on_this_timestep = techs_on_this_timestep.at[tech_on].set(1)
        techs_on_this_timestep *= (tech_on >= 0).astype(
            int
        )  # tech_on could be -1 or -2 for wall or floor so rule this out

        techs_can_achieve = jnp.concatenate((jnp.ones(1), state.completed_techs[:-1]))
        achievable_techs_on_this_timestep = jnp.logical_and(
            techs_on_this_timestep, techs_can_achieve
        )

        techs_keypress = jnp.zeros(TECH_TREE_LENGTH, dtype=jnp.int32)
        techs_keypress = techs_keypress.at[action - 4].set(1)
        techs_keypress = (techs_keypress * (action >= 4)).astype(int)

        achievable_techs_on_this_timestep = jnp.logical_and(
            achievable_techs_on_this_timestep, techs_keypress
        )

        techs_achieved_this_timestep = jnp.logical_and(
            achievable_techs_on_this_timestep, jnp.logical_not(state.completed_techs)
        )
        did_achieve_tech = techs_achieved_this_timestep.astype(int).sum()

        reward_for_tech = ((tech_on + 1) % REWARD_EVERY_N_TECHS) == 0

        reward += (1 * did_achieve_tech * reward_for_tech).astype(int)
        grid = state.grid.at[position[0], position[1]].set(
            (tech_on + 2) * (1 - did_achieve_tech)
        )

        new_completed_techs = jnp.logical_or(
            techs_achieved_this_timestep, state.completed_techs
        ).astype(int)

        state = EnvState(
            grid=grid,
            position=position,
            completed_techs=new_completed_techs,
            timestep=state.timestep + 1,
        )
        done = self.is_terminal(state, params)

        info = {
            "discount": self.discount(state, params),
            "completed_techs": state.completed_techs.astype(int).sum(),
        }

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, rng: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        rng, _rng, __rng = jax.random.split(rng, num=3)
        grid = jax.random.uniform(rng, shape=MAP_SIZE)
        grid = (grid > params.wall_gen_threshold).astype(int)

        position = jax.random.randint(
            _rng, shape=(2,), minval=jnp.zeros(2), maxval=jnp.array(MAP_SIZE)
        )

        grid = grid.at[position[0], position[1]].set(0)

        # Make sure the tech blocks don't overlap
        def _add_tech(carry, unused):
            tech_grid, rng, tech_num = carry

            p = (tech_grid <= 1).astype(float)
            p /= p.sum()
            p = p.flatten()

            num_indexes = MAP_SIZE[0] * MAP_SIZE[1]
            indexes = jnp.arange(num_indexes)

            rng, _rng = jax.random.split(rng)
            index = jax.random.choice(_rng, indexes, p=p)

            index_row = index // tech_grid.shape[0]
            index_col = index % tech_grid.shape[0]

            tech_grid = tech_grid.at[index_row, index_col].set(tech_num)

            return (tech_grid, rng, tech_num + 1), None

        init_carry = (grid, __rng, 2)

        (grid, _, _), _ = jax.lax.scan(
            _add_tech, init_carry, None, length=TECH_TREE_LENGTH
        )

        state = EnvState(
            grid=grid,
            position=position,
            completed_techs=jnp.zeros(TECH_TREE_LENGTH, dtype=jnp.int32),
            timestep=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        shifted_grid = state.grid + 1

        if FULLY_OBSERVABLE:
            # 0 corresponds to the player position
            obs = shifted_grid.at[state.position[0], state.position[1]].set(0)

        else:
            # 0 corresponds to OOB
            padded_grid = jnp.pad(shifted_grid, OBS_DIM // 2 + 2, constant_values=0)

            tl_corner = state.position + 2 * jnp.ones(2, dtype=jnp.int32)

            obs = jax.lax.dynamic_slice(padded_grid, tl_corner, (OBS_DIM, OBS_DIM))

        grid_one_hot = jax.nn.one_hot(obs, num_classes=TECH_TREE_LENGTH + 3)

        repeated_techs = jnp.tile(state.completed_techs, TECH_TREE_OBS_REPEAT)

        obs_with_tech_tree = jnp.concatenate(
            (grid_one_hot.flatten(), repeated_techs), axis=0
        )

        return obs_with_tech_tree

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check termination criteria
        finished_tech_tree = (
            state.completed_techs == jnp.ones_like(state.completed_techs)
        ).all()

        in_wall = state.grid[state.position[0], state.position[1]] == 1

        # Check number of steps in episode termination condition
        done_steps = state.timestep >= params.max_timesteps

        done = jnp.logical_or(finished_tech_tree, in_wall)
        done = jnp.logical_or(done, done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "Gridworld-Tech-Tree-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4 + TECH_TREE_LENGTH

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(4 + TECH_TREE_LENGTH)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        num_classes = TECH_TREE_LENGTH + 3  # (floor, wall, out of bounds/player)
        if FULLY_OBSERVABLE:
            obs_dim = MAP_SIZE[0] * MAP_SIZE[1] * num_classes
        else:
            obs_dim = OBS_DIM * OBS_DIM * num_classes
        return spaces.Box(
            0.0,
            1.0,
            (obs_dim + TECH_TREE_LENGTH * TECH_TREE_OBS_REPEAT,),
            dtype=jnp.int32,
        )

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""

        return spaces.Dict(
            {
                "grid": spaces.Box(0, 1, MAP_SIZE, jnp.int32),
                "position": spaces.Box(0, jnp.max(MAP_SIZE), (2,), jnp.int32),
                "completed_techs": spaces.Box(0, 1, (TECH_TREE_LENGTH,), jnp.int32),
                "timestep": spaces.Box(0, params.max_timesteps, (), jnp.int32),
            }
        )


def obs_to_grid(obs):
    obs_grid = obs[: -TECH_TREE_LENGTH * TECH_TREE_OBS_REPEAT]
    grid = jnp.reshape(obs_grid, (OBS_DIM, OBS_DIM, TECH_TREE_LENGTH + 3))
    return grid
