import heapq

# Action mapping:
# 0: NOOP, 1: LEFT, 2: RIGHT, 3: UP, 4: DOWN, 5: DO
# Movement deltas for actions 1-4:
MOVES = {
    1: (0, -1),  # LEFT:  move left (decrease column)
    2: (0,  1),  # RIGHT: move right (increase column)
    3: (-1, 0),  # UP:    move up (decrease row)
    4: (1,  0)   # DOWN:  move down (increase row)
}

def is_traversable(tile):
    """Return True if the tile is open for planning (tile 2: GRASS)."""
    return tile == 2 or tile == 13

def a_star(grid, start, goal):
    """
    Standard A* search on positions (ignoring orientation).
    Returns a list of actions from start to goal, or None if no path exists.
    """
    rows, cols = len(grid), len(grid[0])
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}  # mapping: pos -> (previous_pos, action)
    cost_so_far = {start: 0}
    
    def heuristic(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for action, (dr, dc) in MOVES.items():
            nr, nc = current[0] + dr, current[1] + dc
            next_pos = (nr, nc)
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if not is_traversable(grid[nr][nc]):
                continue
            new_cost = cost_so_far[current] + 1
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + heuristic(next_pos)
                heapq.heappush(frontier, (priority, next_pos))
                came_from[next_pos] = (current, action)
                
    if goal not in came_from and start != goal:
        return None
    # Reconstruct the path.
    actions = []
    cur = goal
    while cur != start:
        prev, action = came_from[cur]
        actions.append(action)
        cur = prev
    actions.reverse()
    return actions

def multi_target_search(grid, start):
    """
    Computes paths from start to every reachable cell using a Dijkstra-like search.
    Returns:
      - came_from: mapping each cell to (previous_cell, action)
      - cost_so_far: mapping each cell to its cost from start.
    """
    rows, cols = len(grid), len(grid[0])
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while frontier:
        cost, current = heapq.heappop(frontier)
        for action, (dr, dc) in MOVES.items():
            nr, nc = current[0] + dr, current[1] + dc
            next_pos = (nr, nc)
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if not is_traversable(grid[nr][nc]):
                continue
            new_cost = cost_so_far[current] + 1
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                heapq.heappush(frontier, (new_cost, next_pos))
                came_from[next_pos] = (current, action)
    return came_from, cost_so_far

def reconstruct_path(came_from, start, goal):
    """
    Reconstructs the list of actions from start to goal using the came_from mapping.
    """
    if start == goal:
        return []
    actions = []
    cur = goal
    while cur != start:
        prev, action = came_from[cur]
        actions.append(action)
        cur = prev
    actions.reverse()
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
    rows, cols = len(grid), len(grid[0])
    came_from, cost_so_far = multi_target_search(grid, start)
    
    candidate_plans = []  # Each entry: (total_length, plan, candidate_cell, object_cell)
    
    # Find all object cells (e.g. trees have tile value object_id) and sort by Manhattan distance.
    objects = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == object_id:
                dist = abs(r - start[0]) + abs(c - start[1])
                objects.append((dist, r, c))
    objects.sort(key=lambda x: x[0])
    
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
        for (dr, dc), required_orient in rel_moves.items():
            cand = (r + dr, c + dc)
            # Check bounds and candidate traversability.
            if not (0 <= cand[0] < rows and 0 <= cand[1] < cols):
                continue
            if not is_traversable(grid[cand[0]][cand[1]]):
                continue
            if cand not in cost_so_far:
                continue  # candidate not reachable
            path = reconstruct_path(came_from, start, cand)
            final_action = path[-1] if path else None
            if final_action != required_orient:
                # Append the required action to reorient.
                path = path + [required_orient]
            candidate_plans.append((len(path), path, cand, obj))
        
        # Optional early break: if we have found a candidate plan that is as short
        # as the Manhattan distance to the object, further objects are unlikely to yield better plans.
        if candidate_plans:
            best_length = min(candidate_plans, key=lambda x: x[0])[0]
            if best_length <= abs(r - start[0]) + abs(c - start[1]):
                break

    if not candidate_plans:
        print("No reachable object candidate found.")
        return None
    
    best_length, best_plan, best_cand, best_obj = min(candidate_plans, key=lambda x: x[0])
    # Optionally, append the DO action (5) if that is required in your task:
    # best_plan.append(5)
    return best_plan 