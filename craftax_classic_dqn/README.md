# Craftax-Classic DQN

A readable Deep Q-Network baseline for the symbolic Craftax-Classic environment.

The implementation uses a two-layer Flax MLP, replay memory, epsilon-greedy exploration, Huber TD loss, Adam, and a periodically updated target network. It runs headlessly and prints training metrics to the terminal. It does not open the Pygame interface because rendering every environment step would significantly slow training.

## Run

From the repository root:

```bash
python3 craftax_classic_dqn/craftax_classic_dqn.py --smoke-test
```

Train for 100,000 environment steps:

```bash
python3 craftax_classic_dqn/craftax_classic_dqn.py --total-steps 100000
```

Save the final network parameters:

```bash
python3 craftax_classic_dqn/craftax_classic_dqn.py \
  --total-steps 100000 \
  --checkpoint checkpoints/craftax_classic_dqn.msgpack
```

Use `--help` to see all hyperparameters.

## Terminal metrics

- `loss`: Huber temporal-difference loss for the latest optimizer update.
- `q`: mean predicted Q-value for the sampled replay actions.
- `abs_td`: mean absolute temporal-difference error.
- `return_10`: mean return over the latest ten completed training episodes.
- `evaluation return`: mean and standard deviation of greedy-policy returns.
- `score`: terminal Crafter-style achievement score.

For example, `evaluation return=1.767±1.247` means the greedy policy averaged 1.767 reward across the evaluation episodes with a standard deviation of 1.247. The large variation indicates that performance depends strongly on the generated world and that more evaluation episodes and training seeds are needed before drawing a firm conclusion.