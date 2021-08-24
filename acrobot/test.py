import torch
import gym

from acrobot.nnModel import PolicyNN


def test(total_sim=100, steps=500):
    state_dict = torch.load('model.pth')

    model = PolicyNN()
    model.load_state_dict(state_dict)
    model = model.to(device)
    sum_return = 0
    sum_times = []
    for _ in range(total_sim):
        # run this simulation 100 times
        state = env.reset()
        # keep track of what stage we are on
        for i in range(steps):
            res, _ = model.act(state)
            action = res
            state, reward, done, _ = env.step(action)
            if render:
                env.render()
            if done:
                if reward == 1:
                    sum_return += reward
                sum_times.append(i)
                break
        # Keep track of the return
    print(f'Average return: {sum_return / total_sim}')
    print(f'Average times to return: {sum_times / total_sim}')
    if render:
        env.close()


render = False
env = gym.make('Acrobot-v1')
env.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test()