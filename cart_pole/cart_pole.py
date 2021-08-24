# First project using reinforcement learning

import gym
env = gym.make('CartPole-v1')

# basic version, if it is going left, go right, etc
sum_return = 0
for _ in range(100):
    # run this simulation 100 times
    state = env.reset()
    # keep track of what stage we are on
    for i in range(200):
        action = 0 if state[2] <= 0 else 1
        state, reward, done, _ = env.step(action)
        # env.render()
        if done:
            break
    # Keep track of the return
    sum_return += i
print(f'Average return: {sum_return / 100}')
# env.close()

