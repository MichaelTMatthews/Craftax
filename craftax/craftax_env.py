def make_craftax_env_from_name(name: str, auto_reset: bool):
    if auto_reset:
        if name == "Craftax-Symbolic-v1" or name == "Craftax-Symbolic-AutoReset-v1":
            from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv

            return CraftaxSymbolicEnv()
        elif name == "Craftax-Pixels-v1" or name == "Craftax-Pixels-AutoReset-v1":
            from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv

            return CraftaxPixelsEnv()
        if (
            name == "Craftax-Classic-Symbolic-v1"
            or name == "Craftax-Classic-Symbolic-AutoReset-v1"
        ):
            from craftax.craftax_classic.envs.craftax_symbolic_env import (
                CraftaxClassicSymbolicEnv,
            )

            return CraftaxClassicSymbolicEnv()
        elif (
            name == "Craftax-Classic-Pixels-v1"
            or name == "Craftax-Classic-Pixels-AutoReset-v1"
        ):
            from craftax.craftax_classic.envs.craftax_pixels_env import (
                CraftaxClassicPixelsEnv,
            )

            return CraftaxClassicPixelsEnv()
    else:
        if name == "Craftax-Symbolic-v1":
            from craftax.craftax.envs.craftax_symbolic_env import (
                CraftaxSymbolicEnvNoAutoReset,
            )

            return CraftaxSymbolicEnvNoAutoReset()
        elif name == "Craftax-Pixels-v1":
            from craftax.craftax.envs.craftax_pixels_env import (
                CraftaxPixelsEnvNoAutoReset,
            )

            return CraftaxPixelsEnvNoAutoReset()
        elif name == "Craftax-Classic-Symbolic-v1":
            from craftax.craftax_classic.envs.craftax_symbolic_env import (
                CraftaxClassicSymbolicEnvNoAutoReset,
            )

            return CraftaxClassicSymbolicEnvNoAutoReset()
        elif name == "Craftax-Classic-Pixels-v1":
            from craftax.craftax_classic.envs.craftax_pixels_env import (
                CraftaxClassicPixelsEnvNoAutoReset,
            )

            return CraftaxClassicPixelsEnvNoAutoReset()

    raise ValueError(f"Unknown craftax environment: {name}")


def make_craftax_env_from_params(classic: bool, symbolic: bool, auto_reset: bool):
    if classic:
        if symbolic:
            if auto_reset:
                from craftax.craftax_classic.envs.craftax_symbolic_env import (
                    CraftaxClassicSymbolicEnv,
                )

                return CraftaxClassicSymbolicEnv()
            else:
                from craftax.craftax_classic.envs.craftax_symbolic_env import (
                    CraftaxClassicSymbolicEnvNoAutoReset,
                )

                return CraftaxClassicSymbolicEnvNoAutoReset()
        else:
            if auto_reset:
                from craftax.craftax_classic.envs.craftax_pixels_env import (
                    CraftaxClassicPixelsEnv,
                )

                return CraftaxClassicPixelsEnv()
            else:
                from craftax.craftax_classic.envs.craftax_pixels_env import (
                    CraftaxClassicPixelsEnvNoAutoReset,
                )

                return CraftaxClassicPixelsEnvNoAutoReset()
    else:
        if symbolic:
            if auto_reset:
                from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv

                return CraftaxSymbolicEnv()
            else:
                from craftax.craftax.envs.craftax_symbolic_env import (
                    CraftaxSymbolicEnvNoAutoReset,
                )

                return CraftaxSymbolicEnvNoAutoReset()
        else:
            if auto_reset:
                from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv

                return CraftaxPixelsEnv()
            else:
                from craftax.craftax.envs.craftax_pixels_env import (
                    CraftaxPixelsEnvNoAutoReset,
                )

                return CraftaxPixelsEnvNoAutoReset()
