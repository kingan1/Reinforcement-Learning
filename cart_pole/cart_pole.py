# First project using reinforcement learning

import gym
import time
env = gym.make('CartPole-v1')


state = env.reset()
while True:
    action = 1 if state[2] <= 0 else 0
    state, _, done, _ = env.step(action)
    env.render()
    time.sleep(1)
    if done:
        break
env.close()