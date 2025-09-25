import jax
import jax.random
import numpy as np
import gymnasium as gym
from gymnasium import spaces, error
from craftax.craftax_env import make_craftax_env_from_name
from helpers import get_top_down_obs
from typing import List, Sequence  # <-- added
import imageio

class CraftaxTopDownEnv(gym.Env):
    """
    Gymnasium wrapper for Craftax with top-down pixel observations and a *restricted*
    Gymnasium-compatible discrete action space (0..16). Internally maps to raw env actions.

    Custom reward/done:
      - reward_items: list of item names; +1 per unit newly acquired each step
      - done_item: item name whose acquisition (increment) terminates the episode (+1 reward too)
      - include_base_reward: if True, adds original env reward to custom reward

      Special case:
      - If "table" is included in reward_items, then whenever wood decreases by exactly 2
        in a single step, give +1 reward (inventory_increments["table"] = 1 for that step).
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 0}

    def __init__(
        self,
        seed: int | None = None,
        options=None,
        render_mode: str | None = None,
        reward_items: Sequence[str] | None = None,
        done_item: str | None = None,
        include_base_reward: bool = False,
        return_uint8: bool = True,  # <--- added (SB3 friendly)
    ):
        super().__init__()
        self.return_uint8 = return_uint8

        # --- Base Craftax env
        self.env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=True)
        self.env_params = self.env.default_params.replace(
            max_timesteps=100000,
            day_length=99999,
            mob_despawn_distance=0,
        )

        # Custom reward config
        self.reward_items = list(reward_items) if reward_items else []
        self.done_item = done_item
        self.include_base_reward = include_base_reward
        # Optional safeguard:
        # if self.done_item and self.done_item in self.reward_items:
        #     raise ValueError("done_item should not be in reward_items.")

        # --- RNG (JAX) for the underlying env
        if seed is None:
            import secrets
            seed = np.uint32(secrets.randbits(32)).item()
        self._seed = int(seed)
        self.rng = jax.random.PRNGKey(self._seed)

        # --- RNG for Gymnasium API (sampling)
        self.np_random = np.random.default_rng(self._seed)

        # --- Probe shapes and init state
        self.rng, reset_key = jax.random.split(self.rng)
        obs, state = self.env.reset(reset_key, self.env_params)
        top_down = get_top_down_obs(state, obs.copy())
        self.state = state

        if self.return_uint8:
            obs_shape = top_down.shape
            self.observation_space = spaces.Box(
                low=0, high=255, shape=obs_shape, dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=1, shape=top_down.shape, dtype=np.float32
            )

        # --- Action mapping: expose only actions 0..16
        allowed_actions = list(range(17))
        self.allowed_actions = np.asarray(allowed_actions, dtype=np.int32)
        self.action_space = spaces.Discrete(len(self.allowed_actions))

        # Seed the spaces
        self.action_space.seed(self._seed)
        self.observation_space.seed(self._seed)

        # --- Rendering
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode {render_mode}")
        self.render_mode = render_mode

        # --- Episode bookkeeping
        self.elapsed_steps = 0
        self._max_episode_steps = None

        # --- Inventory counters for custom reward
        self._init_inventory_counters()  # sets prev_counts / prev_done_count

    # ---------- Helper methods for custom reward ----------
    def _get_item_count(self, name: str) -> int:
        # Assumes attributes exist on state.inventory; adapt if underlying structure differs
        return int(getattr(self.state.inventory, name))

    def _init_inventory_counters(self):
        self.prev_counts = {item: self._get_item_count(item) for item in self.reward_items}
        if self.done_item:
            self.prev_done_count = self._get_item_count(self.done_item)
        else:
            self.prev_done_count = None

        # Track wood separately for the special "table" reward logic.
        if "table" in self.reward_items:
            try:
                self.prev_wood_count = self._get_item_count("wood")
            except AttributeError:
                self.prev_wood_count = 0
        else:
            self.prev_wood_count = None

    # ---------- Gymnasium API ----------
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._seed = int(seed)
            self.rng = jax.random.PRNGKey(self._seed)
            self.np_random = np.random.default_rng(self._seed)
            self.action_space.seed(self._seed)
            self.observation_space.seed(self._seed)

        self.elapsed_steps = 0

        self.rng, reset_key = jax.random.split(self.rng)
        obs, self.state = self.env.reset(reset_key, self.env_params)
        top_down = get_top_down_obs(self.state, obs.copy())
        if self.return_uint8:
            top_down = (np.clip(top_down, 0, 1) * 255).astype(np.uint8)

        self._init_inventory_counters()

        info = {}
        return top_down, info

    def step(self, action):
        if not np.isscalar(action):
            action = int(np.asarray(action).item())
        if action < 0 or action >= self.action_space.n:
            raise error.InvalidAction(
                f"Action {action} out of range for Discrete({self.action_space.n})"
            )

        raw_action = int(self.allowed_actions[action])

        self.rng, step_key = jax.random.split(self.rng)
        obs, self.state, base_reward, base_done, info = self.env.step(
            step_key, self.state, raw_action, self.env_params
        )
        top_down = get_top_down_obs(self.state, obs.copy())
        if self.return_uint8:
            top_down = (np.clip(top_down, 0, 1) * 255).astype(np.uint8)

        # --- Custom reward calculation
        reward_gain = 0
        item_increments = {}
        for item in self.reward_items:
            # Standard "+1 per unit increase" rule
            new_count = self._get_item_count(item)
            inc = new_count - self.prev_counts[item]
            if inc > 0:
                reward_gain += inc
                item_increments[item] = inc
            self.prev_counts[item] = new_count

        # Special "table" reward: +1 if wood decreased by exactly 2 in this step.
        if "table" in self.reward_items:
            new_wood = self._get_item_count("wood")
            if self.prev_wood_count is None:
                self.prev_wood_count = new_wood
            wood_delta = new_wood - self.prev_wood_count
            if wood_delta == -2:
                reward_gain += 1
                # Expose the event in increments for visibility
                item_increments["table"] = 1
            self.prev_wood_count = new_wood

        terminated = False
        done_increment = 0
        if self.done_item:
            new_done = self._get_item_count(self.done_item)
            done_increment = new_done - self.prev_done_count
            if done_increment > 0:
                reward_gain += done_increment  # +1 (or more if multiple crafted) for done item
                terminated = True
                self.prev_done_count = new_done  # Not strictly needed after termination

        # Optionally add original env reward
        if self.include_base_reward:
            reward_gain += float(base_reward)

        self.elapsed_steps += 1
        truncated = (
            self._max_episode_steps is not None
            and self.elapsed_steps >= self._max_episode_steps
        )

        # Augment info
        info = dict(info)
        info["base_reward"] = float(base_reward)
        info["base_done"] = bool(base_done)
        info["inventory_increments"] = item_increments
        if self.done_item:
            info["done_item_increment"] = done_increment

        return top_down, float(reward_gain), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return None
        elif self.render_mode == "human":
            return None

    def close(self):
        pass


# ---------------- Example usage ----------------
if __name__ == "__main__":
    env = CraftaxTopDownEnv(
        render_mode="rgb_array",
        reward_items=["wood"],
        done_item="wood",
        include_base_reward=False,
        return_uint8=True,  # ensure SB3 compatibility
    )


    obs, info = env.reset(seed=1)

    print("Initial observation:", obs.shape, obs.dtype, obs.max(), obs.min())

    all_obs = [obs.copy()]

    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        all_obs.append(obs.copy())
        total_reward += r
    print("Episode finished with total reward:", total_reward)

    frames = [f for f in all_obs]
    imageio.mimsave(f"craftax_new_env_{total_reward}.gif", frames, fps=5)