import heapq
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_astar_log_level(level):
    """
    Set the logging level for A* pathfinding functions.
    
    Args:
        level: Logging level (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR)
    """
    logger.setLevel(level)
    # Also set the level for the root logger to ensure consistent output
    logging.getLogger().setLevel(level)

def set_astar_verbose(verbose=True):
    """
    Convenience function to set verbose logging for A* pathfinding.
    
    Args:
        verbose: If True, set to DEBUG level. If False, set to WARNING level.
    """
    if verbose:
        set_astar_log_level(logging.DEBUG)
    else:
        set_astar_log_level(logging.WARNING)

def _convert_jax_value(value):
    """
    Convert JAX array values to Python native types for comparison operations.
    
    Args:
        value: A value that might be a JAX array or native Python type
        
    Returns:
        The value converted to a native Python type
    """
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception as e:
            logger.warning(f"Failed to convert JAX value {value} to native type: {e}")
            # Fallback: try to get the value as a scalar
            if hasattr(value, 'tolist'):
                try:
                    return value.tolist()
                except:
                    pass
            return value
    return value

# Action mapping:
# 0: NOOP, 1: LEFT, 2: RIGHT, 3: UP, 4: DOWN, 5: DO
# Movement deltas for actions 1-4:
MOVES = {
    1: (0, -1),  # LEFT:  move left (decrease column)
    2: (0,  1),  # RIGHT: move right (increase column)
    3: (-1, 0),  # UP:    move up (decrease row)
    4: (1,  0)   # DOWN:  move down (increase row)
}

# Block types that can be mined
MINABLE_BLOCKS = {4, 5, 8, 9, 10}  # STONE, TREE, COAL, IRON, DIAMOND

def is_traversable(tile):
    """Return True if the tile is open for planning (tile 2: GRASS)."""
    tile = _convert_jax_value(tile)
    logger.debug(f"Checking if tile {tile} (type: {type(tile)}) is traversable")
    return tile == 2 or tile == 13

def is_minable(tile):
    """Return True if the tile can be mined to become traversable."""
    tile = _convert_jax_value(tile)
    logger.debug(f"Checking if tile {tile} (type: {type(tile)}) is minable")
    return tile in MINABLE_BLOCKS

def a_star_with_mining(grid, start, goal):
    """
    A* search that can mine blocks to create paths.
    Returns a list of actions from start to goal, or None if no path exists.
    The actions include movement actions and mining actions (DO) when needed.
    """
    logger.info(f"Starting A* with mining from {start} to {goal}")
    logger.info(f"Grid dimensions: {len(grid)}x{len(grid[0])}")
    
    # Validate inputs
    if not (0 <= start[0] < len(grid) and 0 <= start[1] < len(grid[0])):
        logger.error(f"Start position {start} is out of bounds for grid size {len(grid)}x{len(grid[0])}")
        return None
    
    if not (0 <= goal[0] < len(grid) and 0 <= goal[1] < len(grid[0])):
        logger.error(f"Goal position {goal} is out of bounds for grid size {len(grid)}x{len(grid[0])}")
        return None
    
    # Check if start and goal are traversable or minable
    start_tile = grid[start[0]][start[1]]
    goal_tile = grid[goal[0]][goal[1]]
    
    if not (is_traversable(start_tile) or is_minable(start_tile)):
        logger.error(f"Start position {start} has non-traversable, non-minable tile: {start_tile}")
        return None
    
    if not (is_traversable(goal_tile) or is_minable(goal_tile)):
        logger.error(f"Goal position {goal} has non-traversable, non-minable tile: {goal_tile}")
        return None
    
    rows, cols = len(grid), len(grid[0])
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}  # mapping: pos -> (previous_pos, action, needs_mining)
    cost_so_far = {start: 0}
    mining_actions = {}  # Track which positions need mining actions
    
    def heuristic(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    nodes_explored = 0
    max_iterations = rows * cols * 2  # Prevent infinite loops
    
    while frontier and nodes_explored < max_iterations:
        _, current = heapq.heappop(frontier)
        nodes_explored += 1
        
        if current == goal:
            logger.info(f"Goal reached after exploring {nodes_explored} nodes")
            break
            
        for action, (dr, dc) in MOVES.items():
            nr, nc = current[0] + dr, current[1] + dc
            next_pos = (nr, nc)
            
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
                
            tile_type = grid[nr][nc]
            
            # Skip if not traversable and not minable
            if not (is_traversable(tile_type) or is_minable(tile_type)):
                logger.debug(f"Skipping position {next_pos} with non-traversable, non-minable tile: {tile_type}")
                continue
                
            # Calculate cost - mining costs extra
            base_cost = 1
            if is_minable(tile_type):
                base_cost = 2  # Mining takes more effort than walking
                
            new_cost = cost_so_far[current] + base_cost
            
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + heuristic(next_pos)
                heapq.heappush(frontier, (priority, next_pos))
                
                # Store whether this position needs mining
                needs_mining = is_minable(tile_type)
                came_from[next_pos] = (current, action, needs_mining)
                
                if needs_mining:
                    mining_actions[next_pos] = True
                    logger.debug(f"Position {next_pos} marked for mining (tile: {tile_type})")
    
    if nodes_explored >= max_iterations:
        logger.warning(f"Max iterations ({max_iterations}) reached, pathfinding may be incomplete")
    
    if goal not in came_from and start != goal:
        logger.error(f"No path found from {start} to {goal} after exploring {nodes_explored} nodes")
        logger.error(f"Frontier size: {len(frontier)}, Cost so far size: {len(cost_so_far)}")
        return None
        
    # Reconstruct the path with mining actions
    actions = []
    cur = goal
    mining_count = 0
    
    while cur != start:
        prev, action, needs_mining = came_from[cur]
        
        # If this position needed mining, add the mining action first
        if needs_mining:
            actions.append(5)  # DO action for mining
            mining_count += 1
            logger.debug(f"Adding mining action for position {cur}")
            
        actions.append(action)
        cur = prev
        
    actions.reverse()
    
    logger.info(f"Path found with {len(actions)} actions ({mining_count} mining actions)")
    logger.debug(f"Final path actions: {actions}")
    
    return actions

def a_star(grid, start, goal):
    """
    Standard A* search on positions (ignoring orientation).
    Returns a list of actions from start to goal, or None if no path exists.
    """
    logger.info(f"Starting standard A* from {start} to {goal}")
    logger.info(f"Grid dimensions: {len(grid)}x{len(grid[0])}")
    
    # Validate inputs
    if not (0 <= start[0] < len(grid) and 0 <= start[1] < len(grid[0])):
        logger.error(f"Start position {start} is out of bounds for grid size {len(grid)}x{len(grid[0])}")
        return None
    
    if not (0 <= goal[0] < len(grid) and 0 <= goal[1] < len(grid[0])):
        logger.error(f"Goal position {goal} is out of bounds for grid size {len(grid)}x{len(grid[0])}")
        return None
    
    # Check if start and goal are traversable
    start_tile = grid[start[0]][start[1]]
    goal_tile = grid[goal[0]][goal[1]]
    
    if not is_traversable(start_tile):
        logger.error(f"Start position {start} has non-traversable tile: {start_tile}")
        return None
    
    if not is_traversable(goal_tile):
        logger.error(f"Goal position {goal} has non-traversable tile: {goal_tile}")
        return None
    
    rows, cols = len(grid), len(grid[0])
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}  # mapping: pos -> (previous_pos, action)
    cost_so_far = {start: 0}
    
    def heuristic(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    nodes_explored = 0
    max_iterations = rows * cols * 2  # Prevent infinite loops
    
    while frontier and nodes_explored < max_iterations:
        _, current = heapq.heappop(frontier)
        nodes_explored += 1
        
        if current == goal:
            logger.info(f"Goal reached after exploring {nodes_explored} nodes")
            break
            
        for action, (dr, dc) in MOVES.items():
            nr, nc = current[0] + dr, current[1] + dc
            next_pos = (nr, nc)
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            try:
                if not is_traversable(grid[nr][nc]):
                    logger.debug(f"Skipping non-traversable position {next_pos} with tile: {grid[nr][nc]}")
                    continue
                new_cost = cost_so_far[current] + 1
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = (current, action)
            except Exception as e:
                logger.warning(f"Error processing position {next_pos}: {e}")
                continue
    
    if nodes_explored >= max_iterations:
        logger.warning(f"Max iterations ({max_iterations}) reached, pathfinding may be incomplete")
                
    if goal not in came_from and start != goal:
        logger.error(f"No path found from {start} to {goal} after exploring {nodes_explored} nodes")
        logger.error(f"Frontier size: {len(frontier)}, Cost so far size: {len(cost_so_far)}")
        return None
        
    # Reconstruct the path.
    actions = []
    cur = goal
    while cur != start:
        prev, action = came_from[cur]
        actions.append(action)
        cur = prev
    actions.reverse()
    
    logger.info(f"Path found with {len(actions)} actions")
    logger.debug(f"Final path actions: {actions}")
    
    return actions

def multi_target_search(grid, start):
    """
    Computes paths from start to every reachable cell using a Dijkstra-like search.
    Returns:
      - came_from: mapping each cell to (previous_cell, action)
      - cost_so_far: mapping each cell to its cost from start.
    """
    logger.info(f"Starting multi-target search from {start}")
    logger.info(f"Grid dimensions: {len(grid)}x{len(grid[0])}")
    
    # Validate start position
    if not (0 <= start[0] < len(grid) and 0 <= start[1] < len(grid[0])):
        logger.error(f"Start position {start} is out of bounds for grid size {len(grid)}x{len(grid[0])}")
        return {}, {}
    
    start_tile = grid[start[0]][start[1]]
    if not is_traversable(start_tile):
        logger.error(f"Start position {start} has non-traversable tile: {start_tile}")
        return {}, {}
    
    rows, cols = len(grid), len(grid[0])
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    nodes_explored = 0
    max_iterations = rows * cols * 2  # Prevent infinite loops
    
    while frontier and nodes_explored < max_iterations:
        cost, current = heapq.heappop(frontier)
        nodes_explored += 1
        
        for action, (dr, dc) in MOVES.items():
            nr, nc = current[0] + dr, current[1] + dc
            next_pos = (nr, nc)
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            try:
                if not is_traversable(grid[nr][nc]):
                    logger.debug(f"Skipping non-traversable position {next_pos} with tile: {grid[nr][nc]}")
                    continue
                new_cost = cost_so_far[current] + 1
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    heapq.heappush(frontier, (new_cost, next_pos))
                    came_from[next_pos] = (current, action)
            except Exception as e:
                logger.warning(f"Error processing position {next_pos}: {e}")
                continue
    
    if nodes_explored >= max_iterations:
        logger.warning(f"Max iterations ({max_iterations}) reached, search may be incomplete")
    
    logger.info(f"Multi-target search completed, explored {nodes_explored} nodes, reached {len(cost_so_far)} positions")
    return came_from, cost_so_far

def multi_target_search_with_mining(grid, start):
    """
    Computes paths from start to every reachable cell using a Dijkstra-like search,
    including minable blocks.
    Returns:
      - came_from: mapping each cell to (previous_cell, action, needs_mining)
      - cost_so_far: mapping each cell to its cost from start.
    """
    logger.info(f"Starting multi-target search with mining from {start}")
    logger.info(f"Grid dimensions: {len(grid)}x{len(grid[0])}")
    
    # Validate start position
    if not (0 <= start[0] < len(grid) and 0 <= start[1] < len(grid[0])):
        logger.error(f"Start position {start} is out of bounds for grid size {len(grid)}x{len(grid[0])}")
        return {}, {}
    
    start_tile = grid[start[0]][start[1]]
    if not (is_traversable(start_tile) or is_minable(start_tile)):
        logger.error(f"Start position {start} has non-traversable, non-minable tile: {start_tile}")
        return {}, {}
    
    rows, cols = len(grid), len(grid[0])
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    nodes_explored = 0
    max_iterations = rows * cols * 2  # Prevent infinite loops
    mining_positions = 0
    
    while frontier and nodes_explored < max_iterations:
        cost, current = heapq.heappop(frontier)
        nodes_explored += 1
        
        for action, (dr, dc) in MOVES.items():
            nr, nc = current[0] + dr, current[1] + dc
            next_pos = (nr, nc)
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
                
            try:
                tile_type = grid[nr][nc]
                if not (is_traversable(tile_type) or is_minable(tile_type)):
                    logger.debug(f"Skipping non-traversable, non-minable position {next_pos} with tile: {tile_type}")
                    continue
                    
                # Calculate cost - mining costs extra
                base_cost = 1
                if is_minable(tile_type):
                    base_cost = 2  # Mining takes more effort than walking
                    
                new_cost = cost_so_far[current] + base_cost
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    heapq.heappush(frontier, (new_cost, next_pos))
                    
                    needs_mining = is_minable(tile_type)
                    came_from[next_pos] = (current, action, needs_mining)
                    
                    if needs_mining:
                        mining_positions += 1
            except Exception as e:
                logger.warning(f"Error processing position {next_pos}: {e}")
                continue
    
    if nodes_explored >= max_iterations:
        logger.warning(f"Max iterations ({max_iterations}) reached, search may be incomplete")
    
    logger.info(f"Multi-target search with mining completed, explored {nodes_explored} nodes, reached {len(cost_so_far)} positions, {mining_positions} mining positions")
    return came_from, cost_so_far

def reconstruct_path(came_from, start, goal):
    """
    Reconstructs the list of actions from start to goal using the came_from mapping.
    """
    if start == goal:
        logger.debug(f"Start and goal are the same: {start}")
        return []
    
    if goal not in came_from:
        logger.error(f"Goal {goal} not found in came_from mapping")
        return None
    
    actions = []
    cur = goal
    path_length = 0
    max_path_length = 1000  # Prevent infinite loops
    
    while cur != start and path_length < max_path_length:
        if cur not in came_from:
            logger.error(f"Position {cur} not found in came_from mapping during path reconstruction")
            return None
            
        prev, action = came_from[cur]
        actions.append(action)
        cur = prev
        path_length += 1
    
    if path_length >= max_path_length:
        logger.error(f"Path reconstruction exceeded maximum length {max_path_length}, possible circular reference")
        return None
    
    if cur != start:
        logger.error(f"Path reconstruction failed to reach start position {start}, ended at {cur}")
        return None
    
    actions.reverse()
    logger.debug(f"Path reconstructed with {len(actions)} actions from {start} to {goal}")
    return actions

def reconstruct_path_with_mining(came_from, start, goal):
    """
    Reconstructs the list of actions from start to goal using the came_from mapping,
    including mining actions where needed.
    """
    if start == goal:
        logger.debug(f"Start and goal are the same: {start}")
        return []
    
    if goal not in came_from:
        logger.error(f"Goal {goal} not found in came_from mapping")
        return None
    
    actions = []
    cur = goal
    path_length = 0
    max_path_length = 1000  # Prevent infinite loops
    mining_count = 0
    
    while cur != start and path_length < max_path_length:
        if cur not in came_from:
            logger.error(f"Position {cur} not found in came_from mapping during path reconstruction")
            return None
            
        prev, action, needs_mining = came_from[cur]
        
        # If this position needed mining, add the mining action first
        if needs_mining:
            actions.append(5)  # DO action for mining
            mining_count += 1
            logger.debug(f"Adding mining action for position {cur}")
            
        actions.append(action)
        cur = prev
        path_length += 1
    
    if path_length >= max_path_length:
        logger.error(f"Path reconstruction exceeded maximum length {max_path_length}, possible circular reference")
        return None
    
    if cur != start:
        logger.error(f"Path reconstruction failed to reach start position {start}, ended at {cur}")
        return None
    
    actions.reverse()
    logger.debug(f"Path reconstructed with {len(actions)} actions ({mining_count} mining actions) from {start} to {goal}")
    return actions

def plan_to_object(grid, start, object_id):
    """
    Plans a path from start to a cell adjacent to an object (e.g. a tree with tile value 5)
    such that when the agent arrives it can face the object.
    
    For each object cell, the candidate cells and required facing actions are:
      - Above the object: candidate = (r-1, c), required action = DOWN (4)
      - Below the object: candidate = (r+1, c), required action = UP   (3)
      - Left  of the object: candidate = (r, c-1), required action = RIGHT (2)
      - Right of the object: candidate = (r, c+1), required action = LEFT  (1)
    
    This version uses a multi-target search for efficiency and omits any extra
    maneuvers – if the final move of the path does not match the required facing,
    the required action is simply appended. (It will not move if blocked by the object.)
    
    Returns the full action sequence (optionally with a DO action appended) or
    None if no candidate is reachable.
    """
    logger.info(f"Planning path to object {object_id} from {start}")
    logger.info(f"Grid dimensions: {len(grid)}x{len(grid[0])}")
    
    # Validate start position
    if not (0 <= start[0] < len(grid) and 0 <= start[1] < len(grid[0])):
        logger.error(f"Start position {start} is out of bounds for grid size {len(grid)}x{len(grid[0])}")
        return None
    
    start_tile = grid[start[0]][start[1]]
    if not is_traversable(start_tile):
        logger.error(f"Start position {start} has non-traversable tile: {start_tile}")
        return None
    
    rows, cols = len(grid), len(grid[0])
    came_from, cost_so_far = multi_target_search(grid, start)
    
    if not came_from:
        logger.error("Multi-target search failed, cannot plan to object")
        return None
    
    candidate_plans = []  # Each entry: (total_length, plan, candidate_cell, object_cell)
    
    # Find all object cells (e.g. trees have tile value object_id) and sort by Manhattan distance.
    objects = []
    for r in range(rows):
        for c in range(cols):
            if _convert_jax_value(grid[r][c]) == object_id:
                dist = abs(r - start[0]) + abs(c - start[1])
                objects.append((dist, r, c))
    
    if not objects:
        logger.error(f"No objects with ID {object_id} found in grid")
        return None
    
    objects.sort(key=lambda x: x[0])
    logger.info(f"Found {len(objects)} objects with ID {object_id}, closest at distance {objects[0][0]}")
    
    # Mapping: candidate cell relative to object -> required facing action.
    # For example, if candidate is above the object, the agent should face DOWN (4) so the object is in front.
    rel_moves = {
        (-1, 0): 4,  # candidate above object: face DOWN
        (1, 0): 3,   # candidate below object: face UP
        (0, -1): 2,  # candidate left of object: face RIGHT
        (0, 1): 1    # candidate right of object: face LEFT
    }
    
    for _, r, c in objects:
        obj = (r, c)
        logger.debug(f"Evaluating object at {obj}")
        
        for (dr, dc), required_orient in rel_moves.items():
            cand = (r + dr, c + dc)
            # Check bounds and candidate traversability.
            if not (0 <= cand[0] < rows and 0 <= cand[1] < cols):
                logger.debug(f"Candidate {cand} is out of bounds")
                continue
            if not is_traversable(grid[cand[0]][cand[1]]):
                logger.debug(f"Candidate {cand} has non-traversable tile: {grid[cand[0]][cand[1]]}")
                continue
            if cand not in cost_so_far:
                logger.debug(f"Candidate {cand} is not reachable from start")
                continue  # candidate not reachable
                
            path = reconstruct_path(came_from, start, cand)
            if path is None:
                logger.warning(f"Failed to reconstruct path to candidate {cand}")
                continue
                
            final_action = path[-1] if path else None
            if final_action != required_orient:
                # Append the required action to reorient.
                path = path + [required_orient]
                logger.debug(f"Added reorientation action {required_orient} to path to candidate {cand}")
                
            candidate_plans.append((len(path), path, cand, obj))
            logger.debug(f"Added candidate plan: length={len(path)}, candidate={cand}, object={obj}")
        
        # Optional early break: if we have found a candidate plan that is as short
        # as the Manhattan distance to the object, further objects are unlikely to yield better plans.
        if candidate_plans:
            best_length = min(candidate_plans, key=lambda x: x[0])[0]
            if best_length <= abs(r - start[0]) + abs(c - start[1]):
                logger.debug(f"Early break: found plan with length {best_length} <= Manhattan distance {abs(r - start[0]) + abs(c - start[1])}")
                break

    if not candidate_plans:
        logger.error(f"No reachable object candidate found for object {object_id}")
        return None
    
    best_length, best_plan, best_cand, best_obj = min(candidate_plans, key=lambda x: x[0])
    logger.info(f"Best plan found: length={best_length}, candidate={best_cand}, object={best_obj}")
    
    # Optionally, append the DO action (5) if that is required in your task:
    # best_plan.append(5)
    return best_plan

def plan_to_object_with_mining(grid, start, object_id):
    """
    Enhanced version of plan_to_object that can mine blocks to reach the target.
    This allows the agent to mine stone, trees, etc. to create paths to objects.
    """
    logger.info(f"Planning path to object {object_id} with mining from {start}")
    logger.info(f"Grid dimensions: {len(grid)}x{len(grid[0])}")
    
    # Validate start position
    if not (0 <= start[0] < len(grid) and 0 <= start[1] < len(grid[0])):
        logger.error(f"Start position {start} is out of bounds for grid size {len(grid)}x{len(grid[0])}")
        return None
    
    start_tile = grid[start[0]][start[1]]
    if not (is_traversable(start_tile) or is_minable(start_tile)):
        logger.error(f"Start position {start} has non-traversable, non-minable tile: {start_tile}")
        return None
    
    rows, cols = len(grid), len(grid[0])
    came_from, cost_so_far = multi_target_search_with_mining(grid, start)
    
    if not came_from:
        logger.error("Multi-target search with mining failed, cannot plan to object")
        return None
    
    candidate_plans = []  # Each entry: (total_length, plan, candidate_cell, object_cell)
    
    # Find all object cells (e.g. trees have tile value object_id) and sort by Manhattan distance.
    objects = []
    for r in range(rows):
        for c in range(cols):
            if _convert_jax_value(grid[r][c]) == object_id:
                dist = abs(r - start[0]) + abs(c - start[1])
                objects.append((dist, r, c))
    
    if not objects:
        logger.error(f"No objects with ID {object_id} found in grid")
        return None
    
    objects.sort(key=lambda x: x[0])
    logger.info(f"Found {len(objects)} objects with ID {object_id}, closest at distance {objects[0][0]}")
    
    # Mapping: candidate cell relative to object -> required facing action.
    # For example, if candidate is above the object, the agent should face DOWN (4) so the object is in front.
    rel_moves = {
        (-1, 0): 4,  # candidate above object: face DOWN
        (1, 0): 3,   # candidate below object: face UP
        (0, -1): 2,  # candidate left of object: face RIGHT
        (0, 1): 1    # candidate right of object: face LEFT
    }
    
    for _, r, c in objects:
        obj = (r, c)
        logger.debug(f"Evaluating object at {obj}")
        
        for (dr, dc), required_orient in rel_moves.items():
            cand = (r + dr, c + dc)
            # Check bounds and candidate traversability (including minable blocks).
            if not (0 <= cand[0] < rows and 0 <= cand[1] < cols):
                logger.debug(f"Candidate {cand} is out of bounds")
                continue
            if not (is_traversable(grid[cand[0]][cand[1]]) or is_minable(grid[cand[0]][cand[1]])):
                logger.debug(f"Candidate {cand} has non-traversable, non-minable tile: {grid[cand[0]][cand[1]]}")
                continue
            if cand not in cost_so_far:
                logger.debug(f"Candidate {cand} is not reachable from start")
                continue  # candidate not reachable
                
            path = reconstruct_path_with_mining(came_from, start, cand)
            if path is None:
                logger.warning(f"Failed to reconstruct path to candidate {cand}")
                continue
                
            final_action = path[-1] if path else None
            if final_action != required_orient:
                # Append the required action to reorient.
                path = path + [required_orient]
                logger.debug(f"Added reorientation action {required_orient} to path to candidate {cand}")
                
            candidate_plans.append((len(path), path, cand, obj))
            logger.debug(f"Added candidate plan: length={len(path)}, candidate={cand}, object={obj}")
        
        # Optional early break: if we have found a candidate plan that is as short
        # as the Manhattan distance to the object, further objects are unlikely to yield better plans.
        if candidate_plans:
            best_length = min(candidate_plans, key=lambda x: x[0])[0]
            if best_length <= abs(r - start[0]) + abs(c - start[1]):
                logger.debug(f"Early break: found plan with length {best_length} <= Manhattan distance {abs(r - start[0]) + abs(c - start[1])}")
                break

    if not candidate_plans:
        logger.error(f"No reachable object candidate found for object {object_id}")
        return None
    
    best_length, best_plan, best_cand, best_obj = min(candidate_plans, key=lambda x: x[0])
    logger.info(f"Best plan found: length={best_length}, candidate={best_cand}, object={best_obj}")
    
    # Optionally, append the DO action (5) if that is required in your task:
    # best_plan.append(5)
    return best_plan 