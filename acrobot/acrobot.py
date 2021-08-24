"""

Replicating cart-pole-v1

"""
import gym

env = gym.make('Acrobot-v1')
import random

def select_action_(state):
    rand = random.random()
    if rand < 0.33:
        return 1
    elif rand < 0.66:
        return 0
    return -1

render = False

def test(total_sim=100, steps=500 ):
    # basic version, if it is going left, go right, etc
    sum_return = 0
    for _ in range(total_sim):
        # run this simulation 100 times
        state = env.reset()
        # keep track of what stage we are on
        for i in range(steps):
            res = select_action_(state)
            action = res
            state, reward, done, _ = env.step(action)
            if render:
                env.render()
            if reward == 1:
                sum_return += 1
                break
        # Keep track of the return - different for this

    print(f'Average return: {sum_return / total_sim}')
    if render:
        env.close()

test()