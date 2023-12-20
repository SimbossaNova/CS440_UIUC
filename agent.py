import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
        
    def best(self, Q, state):
        # Find the action with the maximum Q-value for the given state
        best_action = np.argmax(Q[state])
        return best_action

    
    def update_n(self, state, action):
        # Increment the count for the state-action pair
        self.N[state][action] += 1

    def update_q(self, s, a, r, s_prime):
        # Correctly implement the Q-learning update formula
        # Calculate alpha
        alpha = self.C / (self.C + self.N[s][a])

        # Find the best action for the next state
        best_a = self.best(self.Q, s_prime)
        
        # Update the Q-value
        self.Q[s][a] += alpha * (r + self.gamma * self.Q[s_prime][best_a] - self.Q[s][a])

    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)

        # TODO - MP12: write your function here

        return utils.RIGHT

    def generate_state(self, environment):
        snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y = environment
    
        # Determine food direction
        food_dir_x = 0 if snake_head_x == food_x else (1 if food_x < snake_head_x else 2)
        food_dir_y = 0 if snake_head_y == food_y else (1 if food_y < snake_head_y else 2)
    
        # Initialize adjoining wall and rock variables
        adjoining_wall_x = 0
        adjoining_wall_y = 0
    
        # Check for walls or rock on the left and right of the snake head
        if snake_head_x == 1 or (rock_x <= snake_head_x <= rock_x + 2 and snake_head_y == rock_y):
            adjoining_wall_x = 1
        elif snake_head_x == self.display_width - 2 or (rock_x <= snake_head_x <= rock_x + 2 and snake_head_y == rock_y):
            adjoining_wall_x = 2
    
        # Check for walls or rock above and below the snake head
        if snake_head_y == 1 or (rock_y <= snake_head_y <= rock_y + 1 and snake_head_x == rock_x):
            adjoining_wall_y = 1
        elif snake_head_y == self.display_height - 2 or (rock_y <= snake_head_y <= rock_y + 1 and snake_head_x == rock_x):
            adjoining_wall_y = 2
    
        # Determine adjoining body segments
        adjoining_body_top = 1 if (snake_head_x, snake_head_y - 1) in snake_body else 0
        adjoining_body_bottom = 1 if (snake_head_x, snake_head_y + 1) in snake_body else 0
        adjoining_body_left = 1 if (snake_head_x - 1, snake_head_y) in snake_body else 0
        adjoining_body_right = 1 if (snake_head_x + 1, snake_head_y) in snake_body else 0
    
        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y,
                adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
