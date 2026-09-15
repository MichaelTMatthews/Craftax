# Repository Review Report: Craftax-Classic

**Review scope:** `craftax/craftax_classic` in the checked-out Craftax repository  
**Package version:** 1.6.1, as declared in `pyproject.toml`  
**Purpose of this report:** Meeting-ready summary of how Craftax-Classic is implemented and how it can be used for a first reinforcement-learning experiment

## Executive summary

I reviewed the Craftax GitHub repository, focusing specifically on the `craftax_classic` implementation. Craftax-Classic is a single-agent, open-world survival and crafting environment implemented entirely in JAX. It follows a Gymnax-style functional API: the environment state and a pseudorandom key are explicit inputs to every transition, and `step` returns a new immutable state rather than modifying an object internally.

The environment has a procedurally generated 64 × 64 world, but the agent only observes a 7 × 9 area centered on itself. It must explore, collect resources, craft tools and weapons, manage health and survival needs, and interact with animals and hostile mobs. This creates a long-horizon, partially observable problem with sparse achievement rewards.

The repository exposes symbolic and pixel observations, each with auto-reset and no-auto-reset variants. For an initial RL experiment, the best entry point is `Craftax-Classic-Symbolic-v1` with auto-reset enabled. Its observation is a 1,345-element vector and its action space contains 17 discrete actions. A small PPO agent with an MLP policy is a practical first baseline because the JAX implementation supports fast compilation and parallel environment execution.

The strongest implementation choices are its functional state model, explicit randomness, fixed-shape data structures, and compatibility with JAX transformations such as `jit`, `vmap`, and `lax.scan`. The most important issues for an RL user are the auto-reset semantics, partial observability, sparse rewards, lack of action masking, and the difference between episodic reward and the Crafter-style achievement score in `info`.

## 1. Repository scope and code organization

The Classic implementation contains approximately 4,000 lines of Python. Its main modules are:

| File | Responsibility |
|---|---|
| `craftax_env.py` | Factory that selects Classic/full, symbolic/pixel, and reset behavior |
| `craftax_classic/constants.py` | Block, action and achievement enumerations; rendering constants and textures |
| `craftax_classic/envs/craftax_state.py` | Immutable state, inventory, mob and parameter data structures |
| `craftax_classic/envs/craftax_symbolic_env.py` | Symbolic observation environment and Gymnax-style API |
| `craftax_classic/envs/craftax_pixels_env.py` | RGB observation environment |
| `craftax_classic/world_gen.py` | Procedural map generation and initial state creation |
| `craftax_classic/game_logic.py` | Transitions, interactions, crafting, movement, mobs, survival and reward |
| `craftax_classic/renderer.py` | Symbolic observation encoder and pixel renderer |
| `craftax_classic/envs/common.py` | Terminal achievement metrics and aggregate score |
| `craftax_classic/play_craftax_classic.py` | Interactive Pygame entry point |

The implementation separates the public environment interface from the game simulation. Both symbolic and pixel environments call the same `generate_world` reset function and `craftax_step` transition function; only their observation renderers differ.

## 2. High-level architecture

The main execution path is:

```text
make_craftax_env_from_name(...)
              │
              ▼
   CraftaxClassic*Env
       │          │
       │ reset    │ step
       ▼          ▼
 generate_world  craftax_step
       │          │
       └────┬─────┘
            ▼
         EnvState
            │
            ▼
 symbolic encoder or pixel renderer
            │
            ▼
        agent observation
```

This is a functional architecture. Conceptually, one transition is:

```text
(random key, old state, action, parameters)
                      ↓
(observation, new state, reward, done, info)
```

`EnvState` is a Flax struct dataclass and therefore a JAX PyTree. Updates use expressions such as `state.replace(...)`, producing a new state. This is important because JAX can compile a pure transition function and efficiently apply it across many environments.

## 3. Public environment API

The recommended construction for a first experiment is:

```python
import jax
from craftax.craftax_env import make_craftax_env_from_name

env = make_craftax_env_from_name(
    "Craftax-Classic-Symbolic-v1",
    auto_reset=True,
)
params = env.default_params

key = jax.random.PRNGKey(0)
key, reset_key = jax.random.split(key)
obs, state = env.reset(reset_key, params)
```

The transition signature is:

```python
next_obs, next_state, reward, done, info = env.step(
    step_key,
    state,
    action,
    params,
)
```

Important differences from the newer Gymnasium API are:

- There is one `done` flag rather than separate `terminated` and `truncated` flags.
- The random key is explicit and must not be reused.
- State is explicit; the environment object does not internally hold the current episode state.
- Arrays returned by the environment are JAX arrays.

## 4. Available environment variants

| Observation type | Auto-reset | Factory call |
|---|---:|---|
| Symbolic | Yes | `Craftax-Classic-Symbolic-v1`, `auto_reset=True` |
| Symbolic | No | `Craftax-Classic-Symbolic-v1`, `auto_reset=False` |
| Pixels | Yes | `Craftax-Classic-Pixels-v1`, `auto_reset=True` |
| Pixels | No | `Craftax-Classic-Pixels-v1`, `auto_reset=False` |

The auto-reset and no-auto-reset classes duplicate a small amount of interface code, but both delegate simulation to the same underlying logic. Symbolic observations should be preferred for a first experiment because they avoid learning visual feature extraction and are substantially cheaper to process.

## 5. State and parameters

The complete environment state includes:

- The full tile map and a mob-occupancy map.
- Player position, direction, health, food, drink and energy.
- Hidden recovery, hunger, thirst and fatigue counters.
- Inventory resources, pickaxes and swords.
- Fixed-size arrays for zombies, cows, skeletons and arrows.
- Positions and ages of growing plants.
- Current light level and timestep.
- A Boolean vector of completed achievements.
- A random key used by rendering effects.

The initial player has health, food, drink and energy values of 9 and an empty inventory. The player starts at the center of the map, position `[32, 32]` under the default configuration.

Parameters are split into two structures:

- `EnvParams` contains dynamic configuration such as the 10,000-step episode limit, day length, mob health and spawn probabilities.
- `StaticEnvParams` contains values that determine array shapes, including the 64 × 64 map and maximum numbers of mobs and plants.

This split is appropriate for JAX: changing static shapes normally causes recompilation, while ordinary environment parameters can remain array values passed through compiled functions.

## 6. Procedural world generation

`generate_world` constructs the map from seeded fractal noise and random sampling. Its main stages are:

1. Set the player spawn at the center.
2. Generate water regions and surrounding sand.
3. Generate mountains, paths and caves.
4. Place coal, iron and rare diamond deposits.
5. Generate forests and lava regions.
6. Force the spawn cell to grass so the initial state is valid.
7. Guarantee a diamond deposit when `always_diamond=True`.
8. Initialize empty fixed-capacity mob and plant structures.

Because reset takes a JAX random key, the same reset key and parameters reproduce the same world. Different keys produce different layouts, making procedural generalization part of the task.

## 7. Action space and interaction model

Craftax-Classic has 17 discrete actions:

| ID | Action | Behavior |
|---:|---|---|
| 0 | `NOOP` | Advance time without an intentional action |
| 1 | `LEFT` | Move left and face left |
| 2 | `RIGHT` | Move right and face right |
| 3 | `UP` | Move up and face up |
| 4 | `DOWN` | Move down and face down |
| 5 | `DO` | Interact with the adjacent tile in the facing direction |
| 6 | `SLEEP` | Begin sleeping when energy is below maximum |
| 7 | `PLACE_STONE` | Place one stone in the facing direction |
| 8 | `PLACE_TABLE` | Place a crafting table using two wood |
| 9 | `PLACE_FURNACE` | Place a furnace using one stone |
| 10 | `PLACE_PLANT` | Plant a sapling on grass |
| 11 | `MAKE_WOOD_PICKAXE` | Craft a wood pickaxe |
| 12 | `MAKE_STONE_PICKAXE` | Craft a stone pickaxe |
| 13 | `MAKE_IRON_PICKAXE` | Craft an iron pickaxe |
| 14 | `MAKE_WOOD_SWORD` | Craft a wood sword |
| 15 | `MAKE_STONE_SWORD` | Craft a stone sword |
| 16 | `MAKE_IRON_SWORD` | Craft an iron sword |

`DO` is context-dependent. It can chop a tree, mine a resource, drink water, eat a ripe plant, or attack a mob. It always targets the cell in front of the player rather than selecting a target separately.

The environment does not expose an action mask. If the policy selects an action without satisfying its conditions—for example, crafting without enough resources—the action generally has no useful effect, but time and survival counters still advance. Learning which actions are currently valid is therefore part of the task.

The crafting progression creates dependencies:

```text
tree → wood → crafting table → wood pickaxe
                              ↓
                     stone and coal mining
                              ↓
                    stone pickaxe → iron
                              ↓
             table + furnace + resources
                              ↓
                    iron pickaxe → diamond
```

Weapons use a parallel wood-to-stone-to-iron progression and increase attack damage.

## 8. Observation spaces

### Symbolic observation

The symbolic environment returns a flat `float32` vector with shape `(1345,)` and values declared in the range `[0, 1]`.

| Component | Size |
|---|---:|
| 7 × 9 local map × 17 one-hot block channels | 1,071 |
| 7 × 9 local map × 4 mob channels | 252 |
| Inventory | 12 |
| Health, food, drink and energy | 4 |
| Facing direction one-hot vector | 4 |
| Light level | 1 |
| Sleeping flag | 1 |
| **Total** | **1,345** |

Inventory and player-stat values are divided by 10. The four mob channels represent zombies, cows, skeletons and arrows.

### Pixel observation

The pixel environment returns a normalized RGB array with shape `(63, 63, 3)`. Each map tile is rendered as 7 × 7 pixels. The height includes the 7 visible map rows plus 2 inventory/status rows:

```text
height = (7 + 2) × 7 = 63
width  = 9 × 7       = 63
```

### Partial observability

The symbolic observation does not include the global map, absolute position, achievement vector, hidden survival counters, or complete mob state. A feed-forward policy must infer some state from recent observations, and it cannot perfectly reconstruct everything. A recurrent model can address this later, but it is reasonable to begin with an MLP baseline.

## 9. Transition logic

The central `craftax_step` function applies one action in the following order:

1. Save the initial achievement and health values for reward calculation.
2. Replace the selected action with `NOOP` if the player is already sleeping.
3. Process crafting.
4. Process `DO` interactions: attacking, mining, drinking and eating.
5. Process block placement.
6. Process movement and facing direction.
7. Move existing mobs and resolve their attacks.
8. Spawn new cows, zombies and skeletons.
9. Update plant growth.
10. Update hunger, thirst, fatigue, energy and health.
11. Cap inventory and health values.
12. Calculate reward.
13. Increment the timestep and update daylight.

The ordering is behaviorally important. For example, existing mobs update after the player acts and may attack during the same transition. While sleeping, player actions are ignored until energy is restored or an attack wakes the player.

Mobs use fixed-capacity arrays with Boolean masks rather than Python lists. This avoids dynamic shapes and makes the logic compatible with JAX compilation. Loops over mobs are implemented with `jax.lax.scan` for the same reason.

## 10. Survival system and day/night cycle

The player has four visible survival attributes: health, food, drink and energy. Hidden counters advance every timestep and periodically reduce the visible attributes:

- Hunger eventually decreases food.
- Thirst decreases drink more quickly than hunger decreases food.
- Fatigue decreases energy while active.
- Sleeping restores energy but does not completely stop hunger or thirst.
- Health recovers when necessities are available and declines when they are not.

Light follows a periodic function controlled by `day_length`, which defaults to 300 steps. Zombie spawning becomes more likely as the light level falls. This creates a relationship between time, visibility, enemy pressure and the decision to seek shelter or sleep.

## 11. Reward, achievements and score

The step reward is:

```text
reward = number of newly completed achievements
         + 0.1 × change in player health
```

Each of the 22 achievements gives `+1` only on first completion because achievements are stored as Boolean flags. Losing one health point gives `-0.1`; recovering one health point gives `+0.1`.

Achievements cover resource collection, tool and weapon crafting, placing structures, survival activities and combat. Examples include collecting wood, placing a table, drinking water, making each pickaxe tier, defeating mobs and collecting diamond.

The `info` dictionary also contains one metric per achievement and a `score`. These achievement metrics are multiplied by 100 and emitted at termination, and `score` is their geometric mean with an offset. This Crafter-style score is an evaluation metric; it is not the reward optimized at each step.

The distinction should be maintained in experiment logging:

- **Episode return:** sum of rewards received by the agent.
- **Achievements completed:** interpretable task coverage.
- **`info["score"]`:** aggregate terminal achievement metric.

## 12. Episode termination and auto-reset behavior

An episode ends when any of the following is true:

- The timestep reaches `max_timesteps`, which defaults to 10,000.
- Player health reaches zero.
- The player enters lava.

The auto-reset wrapper has a detail that directly affects RL targets. On a terminal transition, it returns the terminal reward, `done=True`, and terminal `info`, but replaces the returned observation and state with the reset observation and state of a new episode.

Therefore value targets must be masked with `done`:

```python
target = reward + gamma * (1.0 - done) * next_value
```

The returned state should not be examined to decide whether the previous episode terminated because it may already belong to the next episode.

The no-auto-reset variant avoids this replacement, but continuing to step it after termination can produce invalid episode behavior unless an external wrapper performs resets correctly.

## 13. JAX-specific implementation choices

Several repository choices are specifically intended for high-throughput RL:

- **Pure transitions:** State and randomness are explicit inputs and outputs.
- **Immutable PyTrees:** Flax struct dataclasses allow nested state to work with JAX transformations.
- **Fixed capacities:** Mob and plant arrays keep shapes constant; Boolean masks indicate active entries.
- **Functional conditionals:** `jax.lax.select` is used where ordinary Python branches would conflict with tracing.
- **Compiled loops:** `jax.lax.scan` processes mobs and repeated game operations.
- **Stopped environment gradients:** Observations and state returned from `step_env` use `lax.stop_gradient`.
- **Explicit PRNG splitting:** Each stochastic operation receives a derived key.

This architecture is well suited to parallel rollout collection:

```text
jax.vmap: run many independent environments in parallel
jax.lax.scan: run many timesteps inside one compiled computation
jax.jit: compile rollout and possibly the learning update
```

The first reset or step may be slow because it triggers compilation. Subsequent calls with the same shapes are much faster.

## 14. Strengths observed in the implementation

1. **Clear separation of concerns.** World generation, transitions, state definitions and rendering are in separate modules.
2. **One simulation for both observation modes.** Symbolic and pixel variants share the same game logic.
3. **Reproducibility.** Explicit random keys make seeded resets and transitions reproducible.
4. **High-throughput design.** Fixed shapes and pure functions support accelerators and vectorization.
5. **Interpretable progress metrics.** Achievement flags provide more information than total reward alone.
6. **Appropriate benchmark complexity.** The task combines exploration, survival, crafting and combat without the much larger scope of full Craftax.

## 15. Limitations and practical risks

1. **Sparse, hierarchical reward.** Random exploration rarely discovers the full crafting chain, so early learning may be slow.
2. **Partial observability.** The local view and hidden counters mean a feed-forward observation is not Markov.
3. **Very long default episodes.** A 10,000-step horizon complicates credit assignment and evaluation.
4. **No action masking.** The policy spends probability on actions that are invalid in the current state.
5. **Auto-reset must be handled correctly.** Failing to mask terminal bootstrapping will corrupt value targets.
6. **Metric ambiguity.** Reward, individual achievement percentages and geometric-mean score are different quantities and should not be mixed.
7. **Compilation overhead.** Shape or static-configuration changes may trigger recompilation.
8. **Some code duplication.** Auto-reset and no-auto-reset symbolic/pixel classes repeat interface methods.
9. **Wildcard imports and large game-logic module.** `game_logic.py` is over 1,700 lines, and broad imports make dependencies less explicit.
10. **Testing visibility.** I did not find an automated test directory in this local checkout, so changes to mechanics would benefit from targeted transition and invariant tests.

These are mostly engineering or experimentation concerns rather than reasons not to use the environment.

## 16. Recommended first RL experiment

I recommend beginning with PPO and symbolic observations:

| Component | Suggested starting point |
|---|---|
| Environment | `Craftax-Classic-Symbolic-v1` |
| Reset behavior | `auto_reset=True` |
| Observation | 1,345-element symbolic vector |
| Policy | Categorical distribution over 17 actions |
| Network | MLP with two 256-unit hidden layers |
| Value function | Shared encoder with a scalar value head |
| Parallel environments | 64–256, depending on hardware |
| Rollout length | 128–256 steps |
| Discount | Approximately 0.99 |
| Logged metrics | Return, episode length, each achievement, aggregate score |

PPO is not the simplest algorithm mathematically, but it matches the repository architecture well because on-policy rollouts can be collected from many vectorized JAX environments. DQN is also possible with the symbolic vector and 17 actions, but replay-buffer integration and long-horizon sparse rewards can make the first experiment less straightforward. Tabular Q-learning is not suitable because the effective state space is enormous.

The first milestones should be:

1. Verify reset and random-action rollouts.
2. Vectorize reset and step with `jax.vmap`.
3. Confirm that terminal returns and achievements are logged before reset.
4. Train an MLP PPO agent and verify that it reliably obtains early achievements such as collecting wood and drinking water.
5. Only then consider recurrence, intrinsic motivation, action masking or pixel observations.

## 17. Suggested evaluation protocol

For a credible small experiment:

- Use separate training and evaluation PRNG seeds.
- Report mean and standard deviation across multiple training seeds.
- Evaluate without changing the procedural generation rules.
- Track individual achievement completion rates, not only mean return.
- Record episode length and death cause if instrumentation is added.
- State clearly whether auto-reset or no-auto-reset was used.
- State any changed values such as `max_timesteps`, spawn rates or map size.

This matters because two agents with similar returns may have learned very different behaviors. Achievement coverage makes those differences visible.

## 18. Meeting talking points

### One-minute summary

> I reviewed the Craftax repository with a focus on Craftax-Classic. It is a JAX-native, Gymnax-style environment for survival, exploration and hierarchical crafting. The complete world is 64 by 64, but the agent receives only a 7 by 9 local view, so the problem is partially observable. The recommended symbolic environment produces a 1,345-dimensional vector and uses 17 discrete actions. Internally, reset generates a procedural world and every step functionally transforms an immutable state through crafting, interaction, movement, mob updates, survival updates and reward calculation. Reward is one point for each newly completed achievement plus health-change shaping. The implementation is designed for `jit`, `vmap` and `scan`, which makes PPO with many parallel environments a natural first baseline. The main experimental cautions are sparse rewards, long episodes, invalid actions, partial observability and correct handling of auto-reset terminal transitions.

### Main technical point

The most important architectural idea is that the environment is a pure JAX state transition, not a traditional mutable game object. That design is what enables very fast batched RL experiments after compilation.

### Main experimental point

Use symbolic observations first and evaluate with achievement completion rates as well as return. Aggregate reward alone does not show which parts of the crafting and survival hierarchy the policy has learned.

## 19. Likely meeting questions and short answers

**Why use Craftax-Classic instead of full Craftax?**  
It preserves exploration, survival, crafting and combat while using a smaller action and progression space, making it a more manageable first benchmark.

**Why is it partially observable?**  
The agent sees only a local map window and does not observe all internal state variables or the global map.

**Why choose symbolic observations?**  
They expose structured game information directly and let the experiment test the RL algorithm rather than also testing visual representation learning.

**Why PPO?**  
It works naturally with discrete actions and large batches of parallel JAX environments, and it is a standard baseline for Craftax-style tasks.

**What produces reward?**  
A one-time reward for each newly completed achievement plus 0.1 times the change in health.

**What is the biggest implementation gotcha?**  
With auto-reset, the terminal step returns the next episode's initial observation, so learning targets must be masked using `done`.

**Does the agent receive valid-action information?**  
No. It must learn resource and proximity conditions from experience.

**How should progress be measured?**  
Use episodic return, individual achievement completion rates and the aggregate terminal score.

## 20. Implemented DQN baseline and test status

A self-contained DQN baseline is now available in `craftax_classic_dqn/craftax_classic_dqn.py`. It includes:

- A two-layer Flax MLP that predicts Q-values for all 17 actions.
- Epsilon-greedy exploration with linear decay.
- A fixed-size NumPy replay buffer.
- Huber temporal-difference loss and the Adam optimizer.
- A separate target network with periodic hard updates.
- Correct terminal masking in the Bellman target.
- Explicit reset handling for the no-auto-reset environment.
- Greedy evaluation and optional parameter checkpointing.

Run the end-to-end integration test with:

```bash
python3 craftax_classic_dqn/craftax_classic_dqn.py --smoke-test
```

The completed smoke test used 64 environment steps and performed 29 optimizer updates. It exercised replay sampling, gradient updates, target synchronization, episode resets and evaluation, and finished with a finite loss and `SMOKE TEST PASSED`.

A longer 512-step CPU sanity run also completed successfully. It performed 113 optimizer updates and collected nonzero training returns, with a best observed training-episode return of 4.0. Its greedy evaluation return remained zero. This is expected at such a small training budget and should not be presented as evidence that the policy has converged; it shows that the full training loop runs correctly and that the agent receives real Craftax rewards.

Start a larger default run with:

```bash
python3 craftax_classic_dqn/craftax_classic_dqn.py --total-steps 100000
```

To save the final network parameters:

```bash
python3 craftax_classic_dqn/craftax_classic_dqn.py \
  --total-steps 100000 \
  --checkpoint checkpoints/craftax_classic_dqn.msgpack
```

The current implementation is intentionally a readable single-environment baseline. Meaningful performance comparisons require a substantially larger training budget, multiple seeds and individual achievement reporting. Later performance-oriented work could batch environment collection, add Double DQN, use prioritized replay, or add recurrent memory.

## Conclusion

Craftax-Classic is a compact but challenging open-ended RL benchmark. Its game mechanics produce meaningful long-term dependencies, while its functional JAX implementation supports high-throughput experiments. The symbolic auto-reset environment is the appropriate first configuration. A basic MLP PPO agent will provide a useful initial result, although strong performance will likely require better exploration and possibly recurrent memory because the environment is sparse-reward and partially observable.

The repository review suggests that the environment is technically well aligned with JAX-based RL research. Successful experimentation will depend less on basic environment integration and more on careful rollout handling, metric logging, exploration and long-horizon learning.
