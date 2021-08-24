# First project using reinforcement learning

import gym
env = gym.make('CartPole-v1')
render = False
# basic version, if it is going left, go right, etc
sum_return = 0
for _ in range(100):
    # run this simulation 100 times
    state = env.reset()
    # keep track of what stage we are on
    for i in range(200):
        res = state[2] + state[3]
        action = 0 if res <= 0 else 1
        state, reward, done, _ = env.step(action)
        if render:
            env.render()
        if done:
            break
    # Keep track of the return
    sum_return += i
print(f'Average return: {sum_return / 100}')
if render:
    env.close()

