# Installation
```
git clone ...
cd Craftax
pip install -r requirements.txt
pre-commit install
```

Then install jax with GPU enabled

# Run
Run `ppo.py` for experiments, the default arguments should work fine.
To play Craftax run `play_craftax.py` and to play Craftax-Clssic run `play_craftax_classic.py`.
The initial rendering and first 2 movements will trigger JIT compilation and be very slow but after that it should run very fast.