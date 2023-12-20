import heapq

def best_first_search(starting_state):
    visited_states = {starting_state: (None, 0)}
    frontier = []
    heapq.heappush(frontier, starting_state)
    
    while frontier:
        current_state = heapq.heappop(frontier)

        # If the current state is the goal, backtrack to find the path
        if current_state.is_goal():
            return backtrack(visited_states, current_state)

        for neighbor in current_state.get_neighbors():
            if neighbor not in visited_states:
                visited_states[neighbor] = (current_state, current_state.dist_from_start + 1)
                heapq.heappush(frontier, neighbor)

    return []

def backtrack(visited_states, goal_state):
    path = [goal_state]
    current_state = goal_state

    while current_state in visited_states and visited_states[current_state][0] is not None:
        current_state = visited_states[current_state][0]
        path.insert(0, current_state)

    return path
