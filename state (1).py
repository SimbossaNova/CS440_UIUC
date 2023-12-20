from utils import compute_mst_cost
from abc import ABC, abstractmethod

# NOTE: using this global index means that if we solve multiple 
#       searches consecutively the index doesn't reset to 0...
from itertools import count
global_index = count()

# TODO(III): You should read through this abstract class
#           your search implementation must work with this API,
#           namely your search will need to call is_goal() and get_neighbors()
def manhattan(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
class AbstractState(ABC):
    def __init__(self, state, goal, dist_from_start=0, use_heuristic=True):
        self.state = state
        self.goal = goal
        # we tiebreak based on the order that the state was created/found
        self.tiebreak_idx = next(global_index)
        # dist_from_start is classically called "g" when describing A*, i.e., f(state) = g(start, state) + h(state, goal)
        self.dist_from_start = dist_from_start
        self.use_heuristic = use_heuristic
        if use_heuristic:
            self.h = self.compute_heuristic()
        else:
            self.h = 0

    # To search a space we will iteratively call self.get_neighbors()
    # Return a list of AbstractState objects
    @abstractmethod
    def get_neighbors(self):
        pass
    
    # Return True if the state is the goal
    @abstractmethod
    def is_goal(self):
        pass
    
    # A* requires we compute a heuristic from each state
    # compute_heuristic should depend on self.state and self.goal
    # Return a float
    @abstractmethod
    def compute_heuristic(self):
        pass
    
    # The "less than" method ensures that states are comparable, meaning we can place them in a priority queue
    # You should compare states based on f = g + h = self.dist_from_start + self.h
    # Return True if self is less than other
    @abstractmethod
    def __lt__(self, other):
        # NOTE: if the two states (self and other) have the same f value, tiebreak using tiebreak_idx as below
        if self.tiebreak_idx < other.tiebreak_idx:
            return True

    # The "hash" method allow us to keep track of which states have been visited before in a dictionary
    # You should hash states based on self.state (and sometimes self.goal, if it can change)
    # Return a float
    @abstractmethod
    def __hash__(self):
        pass
    # __eq__ gets called during hashing collisions, without it Python checks object equality
    @abstractmethod
    def __eq__(self, other):
        pass

# Grid ------------------------------------------------------------------------------------------------

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
class SingleGoalGridState(AbstractState):
    def __init__(self, state, goal, dist_from_start, use_heuristic, maze_neighbors):
        '''
        state: a length 2 tuple indicating the current location in the grid
        goal: a tuple of a single length 2 tuple location in the grid that needs to be reached, i.e., ((x,y),)
        maze_neighbors(x, y): returns a list of locations in the grid (deals with checking collision with walls, etc.)
        '''
        self.maze_neighbors = maze_neighbors
        super().__init__(state, goal, dist_from_start, use_heuristic)
        
    # TODO(V): implement this method
    def get_neighbors(self):
        nbr_states=[]
        nbr_states = [SingleGoalGridState(loc, self.goal, self.dist_from_start + 1, self.use_heuristic, self.maze_neighbors) 
                    for loc in self.maze_neighbors(*self.state)]
        return nbr_states

    # TODO(V): implement this method, check if the current state is the goal state
    def is_goal(self):
        return self.state == self.goal[0]
    
    def __hash__(self):
        return hash(self.state)
    def __eq__(self, other):
        return self.state == other.state
    
    # TODO(V): implement this method
    # Compute the manhattan distance between self.state and self.goal 
    def compute_heuristic(self):
        return abs(self.state[0] - self.goal[0][0]) + abs(self.state[1] - self.goal[0][1])

    
    # TODO(V): implement this method... should be unchanged from before
    def __lt__(self, other):
        return self.dist_from_start + self.h < other.dist_from_start + other.h or \
               (self.dist_from_start + self.h == other.dist_from_start + other.h and self.tiebreak_idx < other.tiebreak_idx)
    def __str__(self):
        return str(self.state) + ", goal=" + str(self.goal)
    def __repr__(self):
        return str(self.state) + ", goal=" + str(self.goal)



class GridState(AbstractState):
    def __init__(self, state, goal, dist_from_start, use_heuristic, maze_neighbors, mst_cache=None):
        '''
        state: a length 2 tuple indicating the current location in the grid
        goal: a tuple of length 2 tuples location in the grid that needs to be reached
        maze_neighbors(x, y): returns a list of locations in the grid (deals with checking collision with walls, etc.)
        mst_cache: reference to a dictionary which caches a set of goal locations to their MST value
        '''
        self.maze_neighbors = maze_neighbors
        self.mst_cache = mst_cache
        super().__init__(state, goal, dist_from_start, use_heuristic)
        
    
        
    # TODO(VI): implement this method
    def get_neighbors(self):
        nbr_states=[]
        neighboring_locs = self.maze_neighbors(*self.state)
        for loc in neighboring_locs:
            # Update the goals. If the current location matches a goal, remove it from the list of goals
            new_goals = tuple(goal for goal in self.goal if loc != goal)
            nbr_state = GridState(loc, new_goals, self.dist_from_start + 1, self.use_heuristic, self.maze_neighbors, self.mst_cache)
            nbr_states.append(nbr_state)
        return nbr_states

    # TODO(VI): implement this method
    def is_goal(self):
        return len(self.goal) == 0
    
    
    # TODO(VI): implement these methods __hash__ AND __eq__
    def __hash__(self):
        return hash((self.state, self.goal))

    def __eq__(self, other):
        return self.state == other.state and self.goal == other.goal

    
    # TODO(VI): implement this method
    # Our heuristic is: manhattan(self.state, nearest_goal) + MST(self.goal)
    # If we've computed MST(self.goal) before we can simply query the cache, otherwise compute it and cache value
    # NOTE: if self.goal has only one goal then the MST value is simply zero, 
    #       and so the heuristic reduces to manhattan(self.state, self.goal[0])
    # You should use compute_mst_cost(self.goal, manhattan) which we imported from utils.py
    def compute_heuristic(self):
        
        if not self.goal:
            return 0

        nearest_goal_dist = min(abs(self.state[0] - goal[0]) + abs(self.state[1] - goal[1]) for goal in self.goal)

        if len(self.goal) == 1:
            return nearest_goal_dist

        if self.goal in self.mst_cache:
            mst_value = self.mst_cache[self.goal]
        else:
            mst_value = compute_mst_cost(self.goal, lambda x, y: abs(x[0] - y[0]) + abs(x[1] - y[1]))
            self.mst_cache[self.goal] = mst_value

        return nearest_goal_dist + mst_value
    # TODO(VI): implement this method... should be unchanged from before
    def __lt__(self, other):
        return self.dist_from_start + self.h < other.dist_from_start + other.h or \
               (self.dist_from_start + self.h == other.dist_from_start + other.h and self.tiebreak_idx < other.tiebreak_idx)
    # str and repr just make output more readable when your print out states
    def __str__(self):
        return str(self.state) + ", goals=" + str(self.goal)
    def __repr__(self):
        return str(self.state) + ", goals=" + str(self.goal)