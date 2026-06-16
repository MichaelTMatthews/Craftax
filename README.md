<p align="center">
 <img width="80%" src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/logo.png" />
</p>

<p align="center">
        <a href= "https://pypi.org/project/craftax/">
        <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue" /></a>
        <a href= "https://pypi.org/project/craftax/">
        <img src="https://img.shields.io/badge/pypi-1.6.0-green" /></a>
        <a href= "https://pepy.tech/project/craftax">
        <img src="https://static.pepy.tech/badge/craftax" /></a>
       <a href= "https://github.com/MichaelTMatthews/Craftax/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-yellow" /></a>
       <a href= "https://craftaxenv.github.io/">
        <img src="https://img.shields.io/badge/blog-link-purple" /></a>
       <a href= "https://arxiv.org/abs/2402.16801">
        <img src="https://img.shields.io/badge/arxiv-2402.16801-b31b1b" /></a>
<a href= "https://github.com/astral-sh/ruff">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" /></a></p>


# ⛏️ Craftax
Craftax is an RL environment written entirely in <a href="https://github.com/google/jax">JAX</a>.  Craftax reimplements and significantly extends the game mechanics of <a href="https://github.com/danijar/crafter">Crafter</a>, taking inspiration from roguelike games such as <a href="https://github.com/facebookresearch/nle">NetHack</a>.
Craftax conforms to the <a href="https://github.com/RobertTLange/gymnax">gymnax</a> interface, allowing easy integration with existing JAX-based frameworks like <a href="https://github.com/luchris429/purejaxrl">PureJaxRL</a> and [JaxUED](https://github.com/DramaCow/jaxued).

<p align="middle">
  <img src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/archery.gif" width="200" />
  <img src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/building.gif" width="200" /> 
  <img src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/dungeon_crawling.gif" width="200" />
</p>
<p align="middle">
  <img src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/farming.gif" width="200" />
  <img src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/magic.gif" width="200" /> 
  <img src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/mining.gif" width="200" />
</p>

# 📜 Basic Usage
Craftax conforms to the gymnax interface:
```python
rng = jax.random.PRNGKey(0)
rng, _rng = jax.random.split(rng)
_rngs = jax.random.split(_rng, 3)

# Create environment
env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
env_params = env.default_params

# Get an initial state and observation
obs, state = env.reset(_rngs[0], env_params)

# Pick random action
action = env.action_space(env_params).sample(_rngs[1])

# Step environment
obs, state, reward, done, info = env.step(_rngs[2], state, action, env_params)
```

# ⬇️ Installation
The latest Craftax release can be installed from PyPi:
```
pip install craftax
```
If you want the most recent commit instead use:
```
pip install git+https://github.com/MichaelTMatthews/Craftax.git@main
```

## Extending Craftax
If you want to extend Craftax, run (make sure you have `pip>=23.0`):
```
git clone https://github.com/MichaelTMatthews/Craftax.git
cd Craftax
pip install -e ".[dev]"
pre-commit install
```

## GPU-Enabled JAX
By default, both of the above methods will install JAX on the CPU.  If you want to run JAX on a GPU/TPU, you'll need to install the correct wheel for your system from <a href="https://github.com/google/jax?tab=readme-ov-file#installation">JAX</a>.
For NVIDIA GPU the command is:
```
pip install -U "jax[cuda12]"
```

# 🎮 Play
To play Craftax run:
```
play_craftax
```
or to play Craftax-Classic run:
```
play_craftax_classic
```
Since Craftax runs entirely in JAX, it will take some time to compile the rendering and step functions - it might take around 30s to render the first frame and then another 20s to take the first action.  After this it should be very quick.  A tutorial for how to beat the game is present in `tutorial.md`.  The controls are printed out at the beginning of play.

# 📈 Experiment
To run experiments see the [Craftax Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines) repository.

# 🔪 Gotchas
### Optimistic Resets
Craftax provides the option to use optimistic resets to improve performance, which means that we provide access to environments that **do not auto-reset**.
Environments obtained from `make_craftax_env_from_name` or `make_craftax_env_from_args` with `auto_reset=False` will not automatically reset and if not properly handled will continue episodes into invalid states.
These environments should always be wrapped either in `OptimisticResetVecEnvWrapper`(for efficient resets) or `AutoResetEnvWrapper` (to recover the default gymnax auto-reset behaviour).
See `ppo.py` in [Craftax Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines) for correct usage.
Using `auto_reset=True` will return a regular auto-reset environment, which can be treated like any other gymnax environment.

### Texture Caching
We use a texture cache to avoid recreating the texture atlas every time Craftax is imported. If you are just running Craftax as a benchmark this will not affect you.  However, if you are editing the game (e.g. adding new blocks, entities etc.) then a stale cache could cause errors. You can export the following environment variable to force textures to be created from scratch every run.
```
export CRAFTAX_RELOAD_TEXTURES=true
```

# 📋 Scoreboard
If you would like to add an algorithm please open a PR and provide a reference to the source of the results.
We report reward as a % of the maximum (226).
Note that all scores from outside the original Craftax paper are reported and have not been verified.

## Craftax-1B
| Algorithm | Reward (% max) |                                              Code                                               |                  Paper                  |
|:----------|---------------:|:-----------------------------------------------------------------------------------------------:|:---------------------------------------:|
| PPO-GTrXL |           18.3 | [TransformerXL_PPO_JAX](https://github.com/Reytuag/transformerXL_PPO_JAX)                       | [GTrXL](https://arxiv.org/abs/1910.06764)| 
| PQN-RNN   |           16.0 | [purejaxql](https://github.com/mttga/purejaxql/) | [PQN](https://arxiv.org/abs/2407.04811) |
| PPO-RNN   |           15.3 | [Craftax_Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py) | [PPO](https://arxiv.org/abs/1707.06347) |
| RND       |           12.0 | [Craftax_Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnd.py) | [RND](https://arxiv.org/abs/1810.12894) |
| PPO       |           11.9 |   [Craftax_Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo.py)   | [PPO](https://arxiv.org/abs/1707.06347) |
| ICM       |           11.9 |   [Craftax_Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo.py)   | [ICM](https://arxiv.org/pdf/1705.05363) |
| E3B       |           11.0 |   [Craftax_Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo.py)   | [E3B](https://arxiv.org/abs/2210.05805) |


## Craftax-1M
| Algorithm | Reward (% max) |                                              Code                                               |                  Paper                  |
|:----------|---------------:|:-----------------------------------------------------------------------------------------------:|:---------------------------------------:|
| Simulus  |       6.6 |       [Simulus](https://github.com/leor-c/Simulus)                                                    | [Simulus](https://arxiv.org/abs/2502.11537) |
| Efficient MBRL   |            5.4 | - | [Efficient MBRL](https://arxiv.org/abs/2502.01591) |
| PPO-RNN   |            2.3 | [Craftax_Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py) | [PPO](https://arxiv.org/abs/1707.06347) |
| RND       |            2.2 | [Craftax_Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnd.py) | [RND](https://arxiv.org/abs/1810.12894) |
| PPO       |            2.2 |   [Craftax_Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo.py)   | [PPO](https://arxiv.org/abs/1707.06347) |
| ICM       |            2.2 |   [Craftax_Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo.py)   | [ICM](https://arxiv.org/pdf/1705.05363) |
| E3B       |            2.2 |   [Craftax_Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo.py)   | [E3B](https://arxiv.org/abs/2210.05805) |


# 💾 Offline Dataset

A small dataset of mixed-skill human trajectories is available [here](https://drive.google.com/file/d/1wCMdUIsGOWYkNW55Rs0rHkYKUZhaQdtq/view?usp=sharing).
Once the zip file has been extracted, the trajectories can be loaded with the `load_compressed_pickle` function.  These were gathered on an earlier version of Craftax and it is recommended you use [v1.1.0](https://github.com/MichaelTMatthews/Craftax/releases/tag/v1.1.0) or earlier to investigate them.
`run1` is the only trajectory to complete the game.

# ❌ Errata

- Prior to version 1.5.0 there was a bug that made it hard/impossible for the *first* planted plant to ever grow to ripeness in both Craftax and Craftax-Classic. This should have little effect on results as `EAT_PLANT` is an extremely rare achievement, and this only affected the first plant.
- Prior to version 1.6.0 there were two minor bugs for reward on death in Craftax-Classic. Firstly, walking into lava did not give the negative health reward. Secondly, the player could incur extra negative health reward on the final timestep as health could become negative. These may cause minor behavioural differences in trained agents.

# 🔎 See Also
- ⛏️ [Crafter](https://github.com/danijar/crafter) The original Crafter benchmark.
- ⚔️ [NLE](https://github.com/facebookresearch/nle) NetHack as an RL environment.
- ⚡ [PureJaxRL](https://github.com/luchris429/purejaxrl) End-to-end RL implementations in Jax.
- 🌎 [JaxUED](https://github.com/DramaCow/jaxued): CleanRL style UED implementations in Jax.
- 🌍 [Minimax](https://github.com/facebookresearch/minimax): Modular UED implementations in Jax.
- 🏋️ [Gymnax](https://github.com/RobertTLange/gymnax): Standard Jax RL interface with classic environments.
- 🧑‍🤝‍🧑 [JaxMARL](https://github.com/FLAIROx/JaxMARL): Multi-agent RL in Jax.

# 📚 Citation
If you use Craftax in your work please cite it as follows:
```
@inproceedings{matthews2024craftax,
    author={Michael Matthews and Michael Beukman and Benjamin Ellis and Mikayel Samvelyan and Matthew Jackson and Samuel Coward and Jakob Foerster},
    title = {Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning},
    booktitle = {International Conference on Machine Learning ({ICML})},
    year = {2024}
}
```
