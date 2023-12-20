# maze.py
# ---------------
# Licensing Information:  You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Joshua Levine (joshua45@illinois.edu) and Jiaqi Gun

import copy
from state import MazeState, euclidean_distance
from geometry import does_alien_path_touch_wall, does_alien_touch_wall

class MazeError(Exception):
    pass

class NoStartError(Exception):
    pass

class NoObjectiveError(Exception):
    pass

class Maze:
    def __init__(self, alien, walls, waypoints, goals, move_cache={}, k=5, use_heuristic=True):
        self.k = k
        self.alien = alien
        self.walls = walls
        self.states_explored = 0
        self.move_cache = move_cache
        self.use_heuristic = use_heuristic
        self.__start = (*alien.get_centroid(), alien.get_shape_idx())
        self.__objective = tuple(goals)
        self.__waypoints = waypoints + goals
        self.__valid_waypoints = self.filter_valid_waypoints()
        self.__start = MazeState(self.__start, self.get_objectives(), 0, self, self.use_heuristic)

        if not self.__start:
            raise NoStartError("Maze has no start")

        if not self.__objective:
            raise NoObjectiveError("Maze has no objectives")

        if not self.__waypoints:
            raise NoObjectiveError("Maze has no waypoints")

    def is_objective(self, waypoint):
        return waypoint in self.__objective

    def get_start(self):
        assert isinstance(self.__start, MazeState)
        return self.__start

    def set_start(self, start):
        self.__start = start

    def get_objectives(self):
        return copy.deepcopy(self.__objective)

    def get_waypoints(self):
        return self.__waypoints

    def get_valid_waypoints(self):
        return self.__valid_waypoints

    def set_objectives(self, objectives):
        self.__objective = objectives

    def filter_valid_waypoints(self):
        # Initializing a dictionary for valid waypoints
        waypoints_by_shape = {idx: [] for idx in range(len(self.alien.get_shapes()))}
        
        # Create an alien for validation purposes
        validation_alien = self.create_new_alien(0, 0, 0)
        
        # Iterate over alien shapes
        for idx, alien_shape in enumerate(self.alien.get_shapes()):
            # Update validation alien's shape
            validation_alien.set_alien_shape(alien_shape)
            
            # Validate each waypoint
            for point in self.get_waypoints():
                validation_alien.set_alien_pos(point)
                
                # Add waypoint to valid list if alien doesn't intersect with walls
                if not does_alien_touch_wall(validation_alien, self.walls):
                    waypoints_by_shape[idx].append(point)
                    
        return waypoints_by_shape
        
    def get_nearest_waypoints(self, current_waypoint, shape_type):
        distances = []
        
        current_position = (current_waypoint[0], current_waypoint[1], shape_type)
    
        for waypoint in self.filter_valid_waypoints().get(shape_type, []):
            if current_waypoint == waypoint:
                continue
    
            new_position = (waypoint[0], waypoint[1], shape_type)
            if self.is_valid_move(current_position, new_position):
                distance = euclidean_distance(current_position[:2], new_position[:2])
                distances.append((waypoint, distance))
    
        sorted_waypoints = sorted(distances, key=lambda pair: pair[1])
        nearest_waypoints = [pair[0] for pair in sorted_waypoints[:self.k]]
    
        return nearest_waypoints
    
    def create_new_alien(self, x, y, shape_idx):
        alien = copy.deepcopy(self.alien)
        alien.set_alien_config([x, y, self.alien.get_shapes()[shape_idx]])
        return alien

    def is_valid_move(self, start, end):
        def is_out_of_bounds(pos):
            return pos[2] > 2 or pos[2] < 0
    
        def alien_touches_wall_at_position(alien, position, walls):
            alien.set_alien_pos(position[:2])
            return does_alien_touch_wall(alien, walls)
    
        def check_same_shape_path(start, end, walls):
            temp_alien = self.create_new_alien(*start)
            return not does_alien_path_touch_wall(temp_alien, walls, end[:2])
    
        # Check for out of bounds conditions
        if is_out_of_bounds(start) or is_out_of_bounds(end):
            return False
    
        # Check if shapes are the same
        if start[2] == end[2]:
            return check_same_shape_path(start, end, self.walls)
        else:
            if start[:2] != end[:2]:  # Ensure only shape is changed, not position
                return False
            temp_alien = self.create_new_alien(*start[:2], end[2])
            return not alien_touches_wall_at_position(temp_alien, end, self.walls)

    
    def get_neighbors(self, x, y, shape_idx):
        self.states_explored += 1

        nearest = self.get_nearest_waypoints((x, y), shape_idx)
        neighbors = [(*end, shape_idx) for end in nearest]
        for end in [(x, y, shape_idx - 1), (x, y, shape_idx + 1)]:
            start = (x, y, shape_idx)
            if self.is_valid_move(start, end):
                neighbors.append(end)

        return neighbors
