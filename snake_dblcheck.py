from snake_env import SnekEnv

env = SnekEnv()
episodes = 50

for episode in range(episodes):
	done = False
	obs = env.reset()
	while True:#not done:
		random_action = env.action_space.sample()
		#print("action",random_action)
		obs, reward, done, truncated, info = env.step(random_action)
		#print(f'reward :{reward}, done:{done}, truncated:{truncated}, info:{info}')