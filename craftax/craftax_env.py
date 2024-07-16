from craftax.craftax.envs.craftax_pixels_env import (
    CraftaxPixelsEnv,
    CraftaxPixelsEnvNoAutoReset,
)
from craftax.craftax.envs.craftax_symbolic_env import (
    CraftaxSymbolicEnv,
    CraftaxSymbolicEnvNoAutoReset,
)
from craftax.craftax_classic.envs.craftax_pixels_env import (
    CraftaxClassicPixelsEnv,
    CraftaxClassicPixelsEnvNoAutoReset,
)
from craftax.craftax_classic.envs.craftax_symbolic_env import (
    CraftaxClassicSymbolicEnv,
    CraftaxClassicSymbolicEnvNoAutoReset,
)


def make_craftax_env_from_name(name: str, auto_reset: bool):
    if auto_reset:
        if name == "Craftax-Symbolic-v1" or name == "Craftax-Symbolic-AutoReset-v1":
            return CraftaxSymbolicEnv()
        elif name == "Craftax-Pixels-v1" or name == "Craftax-Pixels-AutoReset-v1":
            return CraftaxPixelsEnv()
        if (
            name == "Craftax-Classic-Symbolic-v1"
            or name == "Craftax-Classic-Symbolic-AutoReset-v1"
        ):
            return CraftaxClassicSymbolicEnv()
        elif (
            name == "Craftax-Classic-Pixels-v1"
            or name == "Craftax-Classic-Pixels-AutoReset-v1"
        ):
            return CraftaxClassicPixelsEnv()
    else:
        if name == "Craftax-Symbolic-v1":
            return CraftaxSymbolicEnvNoAutoReset()
        elif name == "Craftax-Pixels-v1":
            return CraftaxPixelsEnvNoAutoReset()
        elif name == "Craftax-Classic-Symbolic-v1":
            return CraftaxClassicSymbolicEnvNoAutoReset()
        elif name == "Craftax-Classic-Pixels-v1":
            return CraftaxClassicPixelsEnvNoAutoReset()

    raise ValueError(f"Unknown craftax environment: {name}")


def make_craftax_env_from_params(classic: bool, symbolic: bool, auto_reset: bool):
    if classic:
        if symbolic:
            if auto_reset:
                return CraftaxClassicSymbolicEnv()
            else:
                return CraftaxClassicSymbolicEnvNoAutoReset()
        else:
            if auto_reset:
                return CraftaxClassicPixelsEnv()
            else:
                return CraftaxClassicPixelsEnvNoAutoReset()
    else:
        if symbolic:
            if auto_reset:
                return CraftaxSymbolicEnv()
            else:
                return CraftaxSymbolicEnvNoAutoReset()
        else:
            if auto_reset:
                return CraftaxPixelsEnv()
            else:
                return CraftaxPixelsEnvNoAutoReset()
