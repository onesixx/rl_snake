# https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
import gymnasium as gym
from gymnasium import spaces

import numpy as np
import cv2
import random
import time
from collections import deque

### Game Rules:
# 1. die 
def collision_with_boundaries(snake_head):
    if snake_head[0]>=500 or snake_head[0]<0 or \
       snake_head[1]>=500 or snake_head[1]<0 :
        return 1
    else:
        return 0
# 2. die 
def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0
    
# 3-1. Score increases and apple is moved to new position
def collision_with_apple(apple_position, score):
    score += 1
    apple_position = [random.randrange(1,50)*10,
                      random.randrange(1,50)*10]
    return apple_position, score

N_DISCRETE_ACTIONS = 4
SNAKE_LEN_GOAL = 300
N_CHANNELS = 5 + SNAKE_LEN_GOAL  
#self.observation =[head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)

env = gym.Env
# env.reset()
class CustomEnv(env):
	def __init__(self, arg1, arg2):      #, ...):
		super(CustomEnv, self).__init__()
		# Define Action and Observation space :: gym.spaces objects 중 하나 

		### Action : Example when using discrete actions
		# self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
		# 0-Left, 1-Right, 3-Up, 2-Down
		self.action_space = spaces.Discrete(4)

		### Obs : Example for using image as input (channel-first; channel-last also works):
		# self.observation_space = spaces.Box(
		# 	low=0, high=255,
		# 	shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8
		# )
		observation = [
			head_x, head_y, 
			apple_delta_x, apple_delta_y, 
			snake_length
		] + list(self.prev_actions)

		### Reward : 
        # self.total_reward = len(self.snake_position) - 3
	def step(self, action):
		...
		return observation, reward, done, info
	
	def reset(self):
		...
		return observation  # reward, done, info can't be included

    # --------------------------------------------------------------------------
	def render(self, mode='human'):
		...

	def close (self):
		...
# ------------------------------------------------------------------------------
env.close()