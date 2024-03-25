import pygame

import jax
import numpy as np

from tech_tree_gridworld.gridworld_tech_tree import (
    Gridworld,
    MAP_SIZE,
    TECH_TREE_LENGTH,
)


class GridworldRenderer:
    def __init__(
        self,
        gridworld: Gridworld,
        env_params,
        gridsquare_render_size: int,
        render_mode: str,
    ):
        self.gridworld = gridworld
        self.env_params = env_params
        self.gridsquare_render_size = gridsquare_render_size
        self.render_mode = render_mode
        self.pygame_events = []

        self.screen_size = (
            MAP_SIZE[0] * gridsquare_render_size,
            MAP_SIZE[1] * gridsquare_render_size,
        )

        # Init rendering
        pygame.init()

        self.screen_surface = pygame.display.set_mode(self.screen_size)

        self.wall_surface = pygame.Surface(
            (gridsquare_render_size, gridsquare_render_size)
        )
        self.wall_surface.fill((128, 128, 128))

        self.player_surface = pygame.Surface(
            (gridsquare_render_size, gridsquare_render_size)
        )
        self.player_surface.fill((64, 64, 255))

        self.tech_surfaces = []
        cols = [
            ((i + 1) * 255 // (TECH_TREE_LENGTH + 1), 0, 0)
            for i in range(TECH_TREE_LENGTH)
        ]
        for tech_num in range(TECH_TREE_LENGTH):
            tech_surface = pygame.Surface(
                (gridsquare_render_size, gridsquare_render_size)
            )
            tech_surface.fill(cols[tech_num])
            self.tech_surfaces.append(tech_surface)

        self.next_tech_surface = pygame.Surface(
            (gridsquare_render_size, gridsquare_render_size)
        )
        self.next_tech_surface.fill((64, 255, 64))

    def render(self, env_state):
        # Update pygame events
        self.pygame_events = list(pygame.event.get())

        # Clear
        self.screen_surface.fill((0, 0, 0))

        # Draw
        grid_squares = []

        grid = np.array(env_state.grid)
        for col in range(MAP_SIZE[0]):
            for row in range(MAP_SIZE[1]):
                x = col * self.gridsquare_render_size
                y = row * self.gridsquare_render_size

                if grid[col, row] == 1:
                    grid_squares.append((self.wall_surface, (x, y)))
                elif grid[col, row] >= 2:
                    tech_num = grid[col, row] - 2
                    tech_surface = self.tech_surfaces[tech_num]
                    if env_state.completed_techs[tech_num] == 0:
                        if (
                            tech_num == 0
                            or env_state.completed_techs[tech_num - 1] == 1
                        ):
                            tech_surface = self.next_tech_surface

                    grid_squares.append((tech_surface, (x, y)))

        self.screen_surface.blits(grid_squares)

        self.screen_surface.blit(
            self.player_surface,
            (
                env_state.position[0] * self.gridsquare_render_size,
                env_state.position[1] * self.gridsquare_render_size,
            ),
        )

        # Update screen
        pygame.display.flip()
        # time.sleep(0.01)

    def is_quit_requested(self):
        for event in self.pygame_events:
            if event.type == pygame.QUIT:
                return True
        return False

    def get_action_from_keypress(self):
        for event in self.pygame_events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    return 0
                elif event.key == pygame.K_RIGHT:
                    return 1
                elif event.key == pygame.K_DOWN:
                    return 2
                elif event.key == pygame.K_LEFT:
                    return 3
                elif event.key == pygame.K_1:
                    return 4
                elif event.key == pygame.K_2:
                    return 5
                elif event.key == pygame.K_3:
                    return 6
                elif event.key == pygame.K_4:
                    return 7
                elif event.key == pygame.K_5:
                    return 8
                elif event.key == pygame.K_6:
                    return 9
                elif event.key == pygame.K_7:
                    return 10
                elif event.key == pygame.K_8:
                    return 11

        return None


def main():
    env = Gridworld()
    env_params = env.default_params

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng = jax.random.split(rng)
    _, env_state = env.reset(_rng, env_params)

    gridsquare_render_size = 32

    render_mode = "state"
    gridworld_renderer = GridworldRenderer(
        env, env_params, gridsquare_render_size, render_mode
    )

    step_fn = jax.jit(env.step)

    while not gridworld_renderer.is_quit_requested():
        action = gridworld_renderer.get_action_from_keypress()

        if action is not None:
            rng, _rng = jax.random.split(rng)
            obs, env_state, reward, done, info = step_fn(
                _rng, env_state, action, env_params
            )
            # print(obs.T)
            if reward > 0:
                print(reward)
            if done:
                print("\n")
        gridworld_renderer.render(env_state)


if __name__ == "__main__":
    debug = True
    if debug:
        with jax.disable_jit():
            main()
    else:
        main()
