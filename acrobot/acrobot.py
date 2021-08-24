"""

Replicating cart-pole-v1

"""
import gym

env = gym.make('Acrobot-v1')

def select_action_(state):
    top_joint = state[2:4]

    s3 = state[4]
    s4 = state[5]
    if s3 + s4 < 0.1:
        # if it's slowing down, go the other way
        if top_joint[1] < 0:
            return -1
        return 1
    else:
        if top_joint[1] < 0:
            return -1
        elif top_joint[1] == 0:
            return 0
        return 1

render = False

def test(total_sim=100, steps=500 ):
    # basic version, if it is going left, go right, etc
    sum_return = 0
    happened = []
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
            if reward == 0:
                sum_return += 1
                break
        # Keep track of the return - different for this

    print(f'Worked correctly: {sum_return / total_sim * 100}%')
    if render:
        env.close()

test()