# train_maskable_ppo.py
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from gymnasium.wrappers import RecordEpisodeStatistics

from top_down_env_gymnasium_options import OptionsOnTopEnv
from top_down_env_gymnasium import CraftaxTopDownEnv  # wherever your class lives

# 1) Build base env (top-down pixels, uint8 for SB3)
base_env = CraftaxTopDownEnv(
    render_mode=None,
    reward_items=[],
    done_item="wood_pickaxe",
    include_base_reward=False,
    return_uint8=True,
)

# 2) Wrap with options layer. Set how many options you have (K).
K = 5  # example: 8 options/skills
env = OptionsOnTopEnv(base_env, num_primitives=16, num_options=K, gamma=0.99, max_skill_len=50)

# (Optional) add stats wrapper
env = RecordEpisodeStatistics(env)

# 3) Create model — MaskablePPO will automatically call env.action_masks()
model = MaskablePPO(
    "CnnPolicy",        # pixels → use CnnPolicy (your obs are image-like uint8)
    env,
    verbose=1,
)

# 4) Evaluate with mask-aware tools (optional)
eval_env = OptionsOnTopEnv(
    CraftaxTopDownEnv(
        render_mode=None,
        reward_items=["wood", "stone", "wood_pickaxe"],
        done_item="stone_pickaxe",
        include_base_reward=False,
        return_uint8=True,
    ),
    num_primitives=16,
    num_options=K,
)
eval_cb = MaskableEvalCallback(
    eval_env,
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
)

# 5) Learn
model.learn(total_timesteps=1_000_000, callback=eval_cb)

all_obs = []
obs, info = eval_env.reset()
all_obs.append(obs.copy())
while True:
    mask = eval_env.action_masks()
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    obs, r, term, trunc, info = eval_env.step(action)
    all_obs.append(obs.copy())
    if term or trunc:
        break
import imageio

frames = [f for f in all_obs]
imageio.mimsave(f"craftax_ppo_options_{r}.gif", frames, fps=5)

print("Eval done.")
print("Last episode reward:", info.get("episode", {}).get("r", None))