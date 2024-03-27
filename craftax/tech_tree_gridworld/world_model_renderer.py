import pygame

import jax.numpy as jnp

from craftax.tech_tree_gridworld.gridworld_tech_tree import (
    Gridworld,
    TECH_TREE_LENGTH,
    OBS_DIM,
    obs_to_grid,
    TECH_TREE_OBS_REPEAT,
)


class WorldModelRenderer:
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
            OBS_DIM * gridsquare_render_size * 4,
            (OBS_DIM + TECH_TREE_OBS_REPEAT) * gridsquare_render_size,
        )

        # Init rendering
        pygame.init()

        self.screen_surface = pygame.display.set_mode(self.screen_size)

        self.wall_surface = pygame.Surface(
            (gridsquare_render_size, gridsquare_render_size)
        )
        self.wall_surface.fill((128, 128, 128))

        self.oob_surface = pygame.Surface(
            (gridsquare_render_size, gridsquare_render_size)
        )
        self.oob_surface.fill((255, 255, 255))

        self.player_surface = pygame.Surface(
            (gridsquare_render_size, gridsquare_render_size)
        )
        self.player_surface.fill((64, 64, 255))
        self.player_surface.set_alpha(127)

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

        self.completed_tech_surface = pygame.Surface(
            (gridsquare_render_size, gridsquare_render_size)
        )
        self.completed_tech_surface.fill((0, 255, 0))

        self.uncompleted_tech_surface = pygame.Surface(
            (gridsquare_render_size, gridsquare_render_size)
        )
        self.uncompleted_tech_surface.fill((255, 0, 0))

    def render(self, env, env_state, next_env_state, collapsed_obs, grad_obs):
        # Update pygame events
        self.pygame_events = list(pygame.event.get())

        # Clear
        self.screen_surface.fill((0, 0, 0))

        self.draw_env_state(env.get_obs(env_state), 0)
        self.draw_env_state(
            env.get_obs(next_env_state), OBS_DIM * self.gridsquare_render_size
        )
        self.draw_env_state(collapsed_obs, OBS_DIM * self.gridsquare_render_size * 2)
        self.draw_grad_state(grad_obs[0], OBS_DIM * self.gridsquare_render_size * 3)

        # Update screen
        pygame.display.flip()
        # time.sleep(0.01)

    def draw_grad_state(self, grad_obs, x_offset):
        grad_obs = jnp.abs(grad_obs)
        grad_obs /= grad_obs.max()

        grid_1h = obs_to_grid(grad_obs)
        grid = jnp.max(jnp.abs(grid_1h), axis=-1)

        min_col = [0, 0, 0]
        max_col = [255, 255, 153]

        def _interpolate_col(c):
            return tuple([int(c * max_col[i] + (1 - c) * min_col[i]) for i in range(3)])

        for col in range(OBS_DIM):
            for row in range(OBS_DIM):
                x = col * self.gridsquare_render_size
                y = row * self.gridsquare_render_size

                color = _interpolate_col(grid[col, row])

                pygame.draw.rect(
                    self.screen_surface,
                    color,
                    pygame.Rect(
                        x + x_offset,
                        y,
                        self.gridsquare_render_size,
                        self.gridsquare_render_size,
                    ),
                )

        # Tech tree
        for tt_repeat in range(TECH_TREE_OBS_REPEAT):
            for tech in range(TECH_TREE_LENGTH):
                grad = grad_obs[
                    -TECH_TREE_LENGTH * TECH_TREE_OBS_REPEAT
                    + tt_repeat * TECH_TREE_LENGTH
                    + tech
                ]

                color = _interpolate_col(grad)

                pygame.draw.rect(
                    self.screen_surface,
                    color,
                    pygame.Rect(
                        (tech) * self.gridsquare_render_size + x_offset,
                        (OBS_DIM + tt_repeat) * self.gridsquare_render_size,
                        self.gridsquare_render_size,
                        self.gridsquare_render_size,
                    ),
                )

    def draw_env_state(self, obs, x_offset):
        completed_techs = obs[-TECH_TREE_LENGTH:]

        # Grid
        grid_squares = []

        grid_1h = obs_to_grid(obs)
        grid = jnp.argmax(grid_1h, axis=-1)

        for col in range(OBS_DIM):
            for row in range(OBS_DIM):
                x = col * self.gridsquare_render_size
                y = row * self.gridsquare_render_size

                if grid[col, row] == 0:
                    grid_squares.append((self.oob_surface, (x + x_offset, y)))
                elif grid[col, row] == 2:
                    grid_squares.append((self.wall_surface, (x + x_offset, y)))
                elif grid[col, row] >= 3:
                    tech_num = grid[col, row] - 3
                    tech_surface = self.tech_surfaces[tech_num]
                    if completed_techs[tech_num] == 0:
                        if tech_num == 0 or completed_techs[tech_num - 1] == 1:
                            tech_surface = self.next_tech_surface

                    grid_squares.append((tech_surface, (x + x_offset, y)))

        self.screen_surface.blits(grid_squares)

        self.screen_surface.blit(
            self.player_surface,
            (
                OBS_DIM // 2 * self.gridsquare_render_size + x_offset,
                OBS_DIM // 2 * self.gridsquare_render_size,
            ),
        )

        # Tech tree
        for tt_repeat in range(TECH_TREE_OBS_REPEAT):
            for tech in range(TECH_TREE_LENGTH):
                tech_completed = obs[
                    -TECH_TREE_LENGTH * TECH_TREE_OBS_REPEAT
                    + tt_repeat * TECH_TREE_LENGTH
                    + tech
                ]
                surface = (
                    self.completed_tech_surface
                    if tech_completed
                    else self.uncompleted_tech_surface
                )

                self.screen_surface.blit(
                    surface,
                    (
                        (tech) * self.gridsquare_render_size + x_offset,
                        (OBS_DIM + tt_repeat) * self.gridsquare_render_size,
                    ),
                )

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
                elif event.key == pygame.K_SPACE:
                    return -1

        return None
