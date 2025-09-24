# options_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces, error

from option_helpers import available_skills, should_terminate, bc_policy

class OptionsOnTopEnv(gym.Env):
    """
    Wraps CraftaxTopDownEnv to expose a Discrete action space:
      [0..P-1]      -> primitive actions (single step)
      [P..P+K-1]    -> options (macro-step via BC until termination)

    MaskablePPO will call `action_masks()` to know which actions are valid.
    Primitives are *always valid*; options are valid iff available_skills(state)[i] is True.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 0}

    def __init__(
        self,
        base_env,                     # an instance of CraftaxTopDownEnv
        num_primitives: int = 17,     # IMPORTANT NOTE says 16 primitive actions
        num_options: int = 0,         # set >0 if you have options
        gamma: float = 0.99,
        max_skill_len: int = 50,      # safety cap for option rollout
    ):
        super().__init__()
        self.env = base_env
        self.gamma = float(gamma)
        self.max_skill_len = int(max_skill_len)

        # ---- Action space mapping
        self.num_primitives = int(num_primitives)
        self.num_options = int(num_options)
        assert self.num_primitives > 0, "Need at least 1 primitive action"

        # Sanity with your underlying CraftaxTopDownEnv mapping
        # (it exposes allowed_actions -> raw actions). We require enough primitives.
        assert self.num_primitives <= self.env.action_space.n, \
            f"num_primitives={self.num_primitives} exceeds base primitive actions ({self.env.action_space.n})"

        # Hybrid action space: primitives first, then options
        self.action_space = spaces.Discrete(self.num_primitives + self.num_options)

        # Observations are just passed through
        self.observation_space = self.env.observation_space

        # Book-keeping
        self.elapsed_macro_steps = 0

        # Seed same as base
        self._seed = getattr(self.env, "_seed", 0)
        self.action_space.seed(self._seed)
        self.observation_space.seed(self._seed)

    # ---------- MaskablePPO hook ----------
    def action_masks(self):
        """
        Returns a bool array of shape (P+K,), True where action is valid *now*.
        - primitives: always True
        - options:    availability from available_skills(self.env.state)
        """
        P, K = self.num_primitives, self.num_options
        prim_mask = np.ones(P, dtype=bool)

        if K == 0:
            return prim_mask  # only primitives

        # Query availability from your helper (expects a state)
        opt_mask_raw = np.asarray(available_skills(self.env.state), dtype=bool)
        if opt_mask_raw.shape[0] != K:
            raise RuntimeError(f"available_skills returned {opt_mask_raw.shape[0]} skills, expected {K}")
        return np.concatenate([prim_mask, opt_mask_raw], axis=0)

    # ---------- Gymnasium API ----------
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.elapsed_macro_steps = 0
        return obs, info

    def step(self, a):
        if not np.isscalar(a):
            a = int(np.asarray(a).item())
        if a < 0 or a >= self.action_space.n:
            raise error.InvalidAction(
                f"Action {a} out of range for Discrete({self.action_space.n})"
            )

        P, K = self.num_primitives, self.num_options

        # --- Case 1: primitive action -> single low-level step
        if a < P:
            # Pass the same index down to the base env (it already maps to raw action)
            obs, r, terminated, truncated, info = self.env.step(a)
            self.elapsed_macro_steps += 1
            return obs, float(r), bool(terminated), bool(truncated), info

        # --- Case 2: option/skill -> run BC until termination
        skill = a - P  # skill id in [0..K-1]
        total_reward = 0.0
        discount = 1.0
        inner_steps = 0
        terminated = False
        truncated = False
        last_info = {}

        while True:
            # 1) query BC controller for the primitive action to execute
            #    BC expects (state, skill) and returns a *primitive* action index in [0..P-1]
            prim_action = int(bc_policy(self.env.state, skill))
            if prim_action < 0 or prim_action >= P:
                raise error.InvalidAction(
                    f"bc_policy returned invalid primitive action {prim_action} (P={P})"
                )

            # 2) step base env once with that primitive
            obs, r, term, trunc, info = self.env.step(prim_action)
            last_info = info
            total_reward += discount * float(r)
            discount *= self.gamma
            inner_steps += 1

            # 3) check termination conditions
            if term or trunc:
                terminated = bool(term)
                truncated = bool(trunc)
                break
            if should_terminate(self.env.state, skill):
                break
            if inner_steps >= self.max_skill_len:
                break

        self.elapsed_macro_steps += 1
        return obs, float(total_reward), terminated, truncated, last_info

    # Optional passthroughs
    def render(self):
        return self.env.render()
    def close(self):
        return self.env.close()