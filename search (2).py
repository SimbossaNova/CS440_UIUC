# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
import heapq
from state import MazeState, euclidean_distance



# Search should return the path and the number of states explored.
# The path should be a list of MazeState objects that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (astar)
# You may need to slight change your previous search functions in MP2 since this is 3-d maze


def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)


# TODO: VI

def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)


# TODO: VI
def astar(maze):
    starting_state = maze.get_start()
    visited_states = {starting_state: (None, 0)}
    closed_set = set()

    frontier = []
    heapq.heappush(frontier, (0, starting_state))

    while frontier:
        _, current_state = heapq.heappop(frontier)
        closed_set.add(current_state)

        for neighbor in current_state.get_neighbors():
            # Skip neighbors that have been closed (already explored)
            if neighbor in closed_set:
                continue

            # Determine the cost to move from current_state to this neighbor
            if current_state.state[2] == neighbor.state[2]:
                move_cost = euclidean_distance(current_state.state, neighbor.state)
            else:
                move_cost = 10

            new_cost = visited_states[current_state][1] + move_cost

            # If this neighbor has not been visited or the new cost is lower
            if neighbor not in visited_states or new_cost < visited_states[neighbor][1]:
                visited_states[neighbor] = (current_state, new_cost)
                total_cost = new_cost + neighbor.h
                heapq.heappush(frontier, (total_cost, neighbor))
                
                # Check for goal when adding to the frontier for efficiency
                if neighbor.is_goal():
                    return backtrack(visited_states, neighbor)

            else:
                # Resurrecting nodes: If the neighbor's cost in the visited_states is greater than its current 
                # dist_from_start, it means we have found a shorter path to this node, so we re-add it to the frontier
                if visited_states[neighbor][1] > neighbor.dist_from_start:
                    visited_states[neighbor] = (current_state, neighbor.dist_from_start)
                    heapq.heappush(frontier, (neighbor.dist_from_start + neighbor.h, neighbor))

    return None


def backtrack(visited_states, goal_state):
    if goal_state not in visited_states:
        return []

    path = [goal_state]
    while visited_states[goal_state][0] is not None:
        goal_state = visited_states[goal_state][0]
        path.append(goal_state)

    return path[::-1]