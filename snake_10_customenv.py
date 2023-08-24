# https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html


import numpy as np
import cv2
import random
import time
from collections import deque

import gymnasium as gym
from gymnasium import spaces

### Game Rules:
# 1. die 
def collision_with_boundaries(snake_head):
    if  snake_head[0]>=500 or snake_head[0]<0 or \
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
    apple_position = [  random.randrange(1,50)*10,
						random.randrange(1,50)*10]
    return apple_position, score

N_DISCRETE_ACTIONS = 4
SNAKE_LEN_GOAL = 30
N_CHANNELS = 5 + SNAKE_LEN_GOAL  # self.observation =[head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)

# env = gym.Env
# env.reset()
class SnekEnv(gym.Env): # CustomEnv(env):
	def __init__(self):      #, ...):
		super(SnekEnv, self).__init__()
		self.np_random = None
		# Define Action and Observation space :: gym.spaces objects 중 하나 

		### Action : Example when using discrete actions
		# self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
		# 0-Left, 1-Right, 3-Up, 2-Down
		self.action_space = spaces.Discrete(4)

		### Obs : Example for using image as input (channel-first; channel-last also works):
		# low / hight:  관찰 가능한 각 차원의 최소값/최대값을 나타내는 배열이나 숫자
		# shape: 상태 공간의 차원을 나타내는 튜플입니다. 예를 들어 (N_CHANNELS, HEIGHT, WIDTH)와 같이 이미지 형태의 공간을 정의할 수 있습니다.
		# dtype: 상태 공간 내의 값들의 데이터 타입을 나타내는 인자로, 보통은 np.float32나 np.uint8과 같은 NumPy 데이터 타입을 사용
		self.observation_space = spaces.Box(
			low=-500, high=500,
			shape=(5+SNAKE_LEN_GOAL, ), dtype=np.float32
		)
		# observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        # self.total_reward = len(self.snake_position) - 3

	def step(self, action):
		self.prev_actions.append(action)
        
		cv2.imshow('snake game', self.img)
		cv2.waitKey(1)
		self.img = np.zeros((500,500,3), dtype='uint8')
        
		# Display Apple
		cv2.rectangle(self.img, 
            (self.apple_position[0],    self.apple_position[1]),
			(self.apple_position[0]+10, self.apple_position[1]+10), (0,255,0), 3)
		# Display Snake
		for position in self.snake_position:
			cv2.rectangle(self.img,
		 		(position[0],   position[1]),
				(position[0]+10,position[1]+10), (0,255,0), 3)

		# Takes step after fixed time
		t_end = time.time() + 0.2      # from 0.05
		k = -1
		while time.time() < t_end:
			if k == -1:
				k = cv2.waitKey(1)
			else:
				continue

		# Change the head position based on the button direction
		button_direction = action
		if button_direction == 1:
			self.snake_head[0] += 10
		elif button_direction == 0:
			self.snake_head[0] -= 10
		elif button_direction == 2:
			self.snake_head[1] += 10
		elif button_direction == 3:
			self.snake_head[1] -= 10
		
		# Increase Snake length on eating apple
		if self.snake_head == self.apple_position:
			self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
			self.snake_position.insert(0, list(self.snake_head))
		else:
			self.snake_position.insert(0, list(self.snake_head))
			self.snake_position.pop()
		
		# On collision kill the snake and print the score
		if collision_with_boundaries(self.snake_head) == 1 or \
		   collision_with_self(self.snake_position) == 1:
			self.done = True

			font = cv2.FONT_HERSHEY_SIMPLEX
			self.img = np.zeros((500,500,3), dtype='uint8')
			cv2.putText(self.img,'Your Score is {}'.format(self.score), (140,250), font, 1, (255,255,255), 2, cv2.LINE_AA)
			cv2.imshow('Game over', self.img)

			# self.reward = -10
		
		# track reward delta and make observation
		self.total_reword = len(self.snake_position) - 3
		self.reward       = self.total_reward - self.prev_reward
		self.prev_reward  = self.total_reward

		if self.done:
			self.reward = -10
		info = {}

		head_x = self.snake_head[0]
		head_y = self.snake_head[1]
	
		snake_length = len(self.snake_position)
		apple_delta_x = self.apple_position[0] - head_x
		apple_delta_y = self.apple_position[1] - head_y

		# create observation:
		observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + \
					list(self.prev_actions)
		observation = np.array(observation)

		return observation, self.reward, self.done, info

	def reset(self, seed=None, options=None):
		if seed is not None:
			self.np_random = np.random.default_rng(seed=seed)
	
		self.img = np.zeros((500, 500,3),dtype='uint8')
		self.snake_head = [250,250]

		self.apple_position = [random.randrange(1,50)*10, random.randrange(1,50)*10]

		self.snake_position = [[250,250],[240,250],[230,250]]
		snake_length = len(self.snake_position)
                
		self.prev_button_direction = 1
		self.button_direction = 1

		self.score = 0    
		self.prev_reward = 0
		self.done = False

		head_x = self.snake_head[0]
		head_y = self.snake_head[1]
		apple_delta_x = self.apple_position[0] - head_x
		apple_delta_y = self.apple_position[1] - head_y

		self.prev_actions = deque(maxlen = SNAKE_LEN_GOAL)  # however long we aspire the snake to be
		for i in range(SNAKE_LEN_GOAL):
			self.prev_actions.append(-1) # to create history

		### create observation:
		observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + \
                	list(self.prev_actions)
		observation = np.array(observation)
		info = {}		
		return observation, info # reward, done, info can't be included
    # --------------------------------------------------------------------------
	# def render(self, mode='human'):
	# 	...

	# def close (self):
	# 	...
# ------------------------------------------------------------------------------
#env.close()