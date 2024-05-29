from craftax.craftax.craftax_state import EnvState
from craftax.craftax.constants import *


def log_achievements_to_info(state: EnvState, done: bool):
    achievements = state.achievements * done * 100.0
    info = {}
    for achievement in Achievement:
        name = f"Achievements/{achievement.name.lower()}"
        info[name] = achievements[achievement.value]
    return info
