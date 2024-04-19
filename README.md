<p align="center">
 <img width="80%" src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/logo.png" />
</p>

<p align="center">
        <a href= "https://pypi.org/project/craftax/">
        <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue" /></a>
        <a href= "https://pypi.org/project/craftax/">
        <img src="https://img.shields.io/badge/pypi-1.3.0-green" /></a>
       <a href= "https://github.com/MichaelTMatthews/Craftax/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-yellow" /></a>
       <a href= "https://craftaxenv.github.io/">
        <img src="https://img.shields.io/badge/blog-link-purple" /></a>
       <a href= "https://arxiv.org/pdf/2402.16801.pdf">
        <img src="https://img.shields.io/badge/arxiv-2402.16801-b31b1b" /></a>
       <a href= "https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
</p>



# ⛏️ Craftax
Craftax is an RL environment written entirely in <a href="https://github.com/google/jax">JAX</a>.  Craftax reimplements and significantly extends the game mechanics of <a href="https://danijar.com/project/crafter/">Crafter</a>, taking inspiration from roguelike games such as <a href="https://github.com/facebookresearch/nle">NetHack</a>.
Craftax conforms to the <a href="https://github.com/RobertTLange/gymnax">gymnax</a> interface, allowing easy integration with existing JAX-based frameworks like <a href="https://chrislu.page/blog/meta-disco/">PureJaxRL</a> and [JaxUED](https://github.com/DramaCow/jaxued).

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
rngs = jax.random.split(_rng, 3)

# Create environment
env = AutoResetEnvWrapper(CraftaxSymbolicEnv())
env_params = env.default_params

# Get an initial state and observation
obs, state = env.reset(rngs[0], env_params)

# Pick random action
action = env.action_space(env_params).sample(rngs[1])

# Step environment
obs, state, reward, done, info = env.step(rngs[2], state, action, env_params)
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
pip install --editable .
```

## GPU-Enabled JAX
By default, both of the above methods will install JAX on the CPU.  If you want to run JAX on a GPU/TPU, you'll need to install the correct wheel for your system from <a href="https://github.com/google/jax?tab=readme-ov-file#installation">JAX</a>.
For NVIDIA GPU the command is:
```
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
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
Craftax provides the option to use optimistic resets to improve performance, which means that (unlike regular gymnax environments) it **does not auto-reset** by default.
This means that the environment should always be wrapped either in `OptimisticResetVecEnvWrapper` (for efficient resets) or `AutoResetEnvWrapper` (to recover the default gymnax auto-reset behaviour).  See `ppo.py` for correct usage of both wrappers.

### Texture Caching
We use a texture cache to avoid recreating the texture atlas every time Craftax is imported. If you are just running Craftax as a benchmark this will not affect you.  However, if you are editing the game (e.g. adding new blocks, entities etc.) then a stale cache could cause errors. You can export the following environment variable to force textures to be created from scratch.
```
export CRAFTAX_RELOAD_TEXTURES=true
```

# 📋 Scoreboard
If you would like to add an algorithm please open a PR and provide a reference to the source of the results.
We report reward as a % of the maximum (226).

## Craftax-1B
| Algorithm | Reward (% max) |                                  Source                                   |
|:----------|---------------:|:-------------------------------------------------------------------------:|
| PPO-RNN   |           15.3 | <a href="https://github.com/luchris429/purejaxrl/tree/main">PureJaxRL</a> |
| PPO       |           11.9 | <a href="https://github.com/luchris429/purejaxrl/tree/main">PureJaxRL</a> |
| ICM       |           11.9 |           <a href="https://arxiv.org/abs/1705.05363">ICM</a>              |
| E3B       |           11.0 |            <a href="https://arxiv.org/abs/2210.05805">E3B</a>             |


## Craftax-1M
| Algorithm | Reward (% max) |                                  Source                                   |
|:----------|---------------:|:-------------------------------------------------------------------------:|
| PPO-RNN   |            2.3 | <a href="https://github.com/luchris429/purejaxrl/tree/main">PureJaxRL</a> |
| PPO       |            2.2 | <a href="https://github.com/luchris429/purejaxrl/tree/main">PureJaxRL</a> |
| ICM       |            2.2 |           <a href="https://arxiv.org/abs/1705.05363">ICM</a>              |
| E3B       |            2.2 |            <a href="https://arxiv.org/abs/2210.05805">E3B</a>             |


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
@article{matthews2024craftax,
  title={Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning},
  author={Michael Matthews and Michael Beukman and Benjamin Ellis and Mikayel Samvelyan and Matthew Jackson and Samuel Coward and Jakob Foerster},
  journal={arXiv preprint},
  year={2024},
}
```