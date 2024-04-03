from craftax.craftax.craftax_state import EnvState
from craftax.craftax.constants import *


def compute_score(state: EnvState, done: bool):
    achievements = state.achievements * done * 100.0
    info = {}
    for achievement in Achievement:
        name = f"Achievements/{achievement.name.lower()}"
        info[name] = achievements[achievement.value]
    # Geometric mean with an offset of 1%
    info["score"] = jnp.exp(jnp.mean(jnp.log(1 + achievements))) - 1.0
    return info
