import jax
import jax.random
import numpy as np
import gymnasium as gym
from gymnasium import spaces, error
from craftax.craftax_env import make_craftax_env_from_name
from helpers import get_top_down_obs


class CraftaxTopDownEnv(gym.Env):
    """
    Gymnasium wrapper for Craftax with top-down pixel observations and a *restricted*
    Gymnasium-compatible discrete action space (0..16). Internally maps to raw env actions.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 0}

    def __init__(self, seed: int | None = None, options=None, render_mode: str | None = None):
        super().__init__()

        # --- Base Craftax env
        self.env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=True)
        self.env_params = self.env.default_params.replace(
            max_timesteps=100000,
            day_length=99999,
            mob_despawn_distance=0,
        )

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

        # --- Observation space (uint8 image)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=top_down.shape, dtype=np.uint8
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

        # --- Episode bookkeeping (for truncation)
        self.elapsed_steps = 0
        self._max_episode_steps = None  # or set to int if you want a hard time limit

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
        top_down = get_top_down_obs(self.state, obs.copy()).astype(np.uint8)

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
        obs, self.state, reward, done, info = self.env.step(
            step_key, self.state, raw_action, self.env_params
        )
        top_down = get_top_down_obs(self.state, obs.copy()).astype(np.uint8)

        terminated = bool(done)
        self.elapsed_steps += 1
        truncated = (
            self._max_episode_steps is not None
            and self.elapsed_steps >= self._max_episode_steps
        )

        return top_down, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            # you could cache last obs in reset/step and return it here
            return None
        elif self.render_mode == "human":
            return None

    def close(self):
        pass


# ---------------- Example usage ----------------
if __name__ == "__main__":
    env = CraftaxTopDownEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        total_reward += r
    print("Episode finished with total reward:", total_reward)