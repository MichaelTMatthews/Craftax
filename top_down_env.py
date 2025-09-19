import jax
import jax.random
import numpy as np
import gym
from craftax.craftax_env import make_craftax_env_from_name
from helpers import get_top_down_obs

class ActionSpaceWrapper:
    """
    A wrapper for the Craftax action space so that the sample() method does not
    require an RNG argument. It internally splits a JAX RNG key for each sample.
    """
    def __init__(self, base_action_space, rng):
        self.base_action_space = base_action_space
        self.rng = rng

    def sample(self):
        self.rng, subkey = jax.random.split(self.rng)
        return self.base_action_space.sample(subkey)

    def __getattr__(self, attr):
        return getattr(self.base_action_space, attr)

class CraftaxTopDownEnv(gym.Env):
    """
    Gym wrapper for the Craftax environment using top-down pixel observations.
    This wrapper removes all data saving and planning logic.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, seed=None):
        super(CraftaxTopDownEnv, self).__init__()
        # Create the Craftax environment with pixel observations.
        self.env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=True)
        self.env_params = self.env.default_params.replace(
            max_timesteps=100000,
            day_length=99999,
            mob_despawn_distance=0,
        )
        
        # Initialize a JAX random key (seeded).
        if seed is None:
            import secrets
            seed = np.uint32(secrets.randbits(32)).item()
        self._seed = int(seed)
        self.rng = jax.random.PRNGKey(self._seed)
        
        # Reset once to determine observation shape.
        self.rng, reset_key = jax.random.split(self.rng)
        obs, state = self.env.reset(reset_key, self.env_params)
        top_down = get_top_down_obs(state, obs.copy())
        
        # Define the Gym observation space (assumes top_down is an image array).
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=top_down.shape, dtype=np.uint8)
        
        # Wrap the Craftax action space so that sample() requires no rng argument.
        self.rng, action_rng = jax.random.split(self.rng)
        raw_action_space = self.env.action_space(self.env_params)
        self.action_space = ActionSpaceWrapper(raw_action_space, action_rng)
        
        # Store the current environment state.
        self.state = state

    def seed(self, seed=None):
        """Set the RNG seed. Returns [seed] for Gym compatibility."""
        if seed is None:
            import secrets
            seed = np.uint32(secrets.randbits(32)).item()
        self._seed = int(seed)
        self.rng = jax.random.PRNGKey(self._seed)
        # refresh action-space RNG too
        self.rng, action_rng = jax.random.split(self.rng)
        self.action_space.rng = action_rng
        return [self._seed]

    def reset(self, seed=None):
        """
        Resets the environment and returns the initial top-down observation.
        Optionally reseeds with a provided seed.
        """
        if seed is not None:
            self.seed(seed)
        self.rng, reset_key = jax.random.split(self.rng)
        obs, self.state = self.env.reset(reset_key, self.env_params)
        top_down = get_top_down_obs(self.state, obs.copy())
        return top_down

    def step(self, action):
        """
        Applies an action to the environment and returns:
          - top-down observation,
          - reward,
          - done flag,
          - and additional info.
        """
        self.rng, step_key = jax.random.split(self.rng)
        obs, self.state, reward, done, info = self.env.step(step_key, self.state, action, self.env_params)
        top_down = get_top_down_obs(self.state, obs.copy())
        return top_down, reward, done, info

    def render(self, mode='human'):
        # Optionally implement rendering logic (e.g., displaying the top-down image).
        pass

    def close(self):
        # Clean up resources if necessary.
        pass

# Example usage: running a single episode with random actions.
if __name__ == "__main__":
    env = CraftaxTopDownEnv()
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Now sample() works without passing an RNG.
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print("Episode finished with total reward:", total_reward)