import torch
import gym
from acrobot.nnModel import PolicyNN

render = False
env = gym.make("Acrobot-v1")
env.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PolicyNN()
model.load_state_dict(torch.load("model.pth"))
model = model.to(device)


def test(total_sim=100, steps=500):
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

    print(f"Average return: {sum(sum_return) / total_sim}")
    print("Negative sum of turns until goal reached")

    if render:
        env.close()


test()
