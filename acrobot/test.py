import torch
import gym

from acrobot.nnModel import PolicyNN


def test(total_sim=100, steps=500):
    state_dict = torch.load('model.pth')

    model = PolicyNN()
    model.load_state_dict(state_dict)
    model = model.to(device)
    sum_return = []
    for _ in range(total_sim):
        # run this simulation 100 times
        state = env.reset()
        # keep track of what stage we are on
        return_time = []
        for i in range(steps):
            res, _ = model.act(state)
            action = res
            state, reward, done, _ = env.step(action)
            return_time.append(reward)
            if render:
                env.render()
            if done:
                break
        sum_return.append(sum(return_time))
        # more like sum up the return values
        # Keep track of the return

    print(f'Average return: {sum(sum_return) / total_sim}')
    print(f'this is a negative number indication how many turns it takes for the goal to be reached')

    if render:
        env.close()


render = False
env = gym.make('Acrobot-v1')
env.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test()