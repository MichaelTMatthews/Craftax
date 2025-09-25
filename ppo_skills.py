# train_maskable_ppo.py
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from gymnasium.wrappers import RecordEpisodeStatistics

from top_down_env_gymnasium_options import OptionsOnTopEnv
from top_down_env_gymnasium import CraftaxTopDownEnv  # wherever your class lives

# NEW: imports for TensorBoard callbacks
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from datetime import datetime

# NEW: a light callback that logs per-episode info fields to TensorBoard
class InfoToTensorboardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self._step_counter = 0

    def _on_step(self) -> bool:
        # "infos" is a list (one per env in the VecEnv)
        infos = self.locals.get("infos", [])
        for info in infos:
            # When episode ends, Gymnasium's RecordEpisodeStatistics adds "episode"
            if isinstance(info, dict) and "episode" in info:
                ep = info["episode"]
                # Standard episode stats
                self.logger.record("episode/reward", float(ep.get("r", np.nan)))
                self.logger.record("episode/length", float(ep.get("l", np.nan)))
                self.logger.record("episode/time", float(ep.get("t", np.nan)))

                # Log any other numeric info keys your env emits at episode end
                # Example keys you might add in your env: "option_switches", "mean_skill_len",
                # "success", "invalid_mask_frac", etc.
                for k, v in info.items():
                    if k == "episode":
                        continue
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        self.logger.record(f"info/{k}", float(v))

        self._step_counter += 1
        return True

# 1) Build base env (top-down pixels, uint8 for SB3)
base_env = CraftaxTopDownEnv(
    render_mode=None,
    reward_items=["wood", "stone", "wood_pickaxe"],
    done_item="stone_pickaxe",
    include_base_reward=False,
    return_uint8=True,
)

# 2) Wrap with options layer. Set how many options you have (K).
K = 5  # example: 8 options/skills
env = OptionsOnTopEnv(base_env, num_primitives=16, num_options=K, gamma=0.99, max_skill_len=50)

# (Optional) add stats wrapper
env = RecordEpisodeStatistics(env)

# NEW: run name + TensorBoard dir
run_name = f"ppo_maskable_options_K{K}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
tb_log_dir = f"./runs/{run_name}"

# 3) Create model — MaskablePPO will automatically call env.action_masks()
model = MaskablePPO(
    "CnnPolicy",        # pixels → use CnnPolicy (your obs are image-like uint8)
    env,
    verbose=1,
    tensorboard_log=tb_log_dir,  # NEW: enable TensorBoard logging
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
    eval_freq=10,
    n_eval_episodes=10,
    deterministic=True,
    # Note: EvalCallback will record eval/mean_reward and eval/mean_ep_length to TB via the logger
)

# NEW: combine callbacks (eval + tensorboard info)
tb_info_cb = InfoToTensorboardCallback()
callbacks = CallbackList([eval_cb, tb_info_cb])

# 5) Learn
model.learn(
    total_timesteps=100,
    callback=callbacks,
    tb_log_name="train",   # TB run subfolder under tb_log_dir
    log_interval=10,
    progress_bar=True
)

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