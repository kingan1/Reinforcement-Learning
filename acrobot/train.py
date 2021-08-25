"""

Training the acrobot model

"""

import gym
import torch
import torch.optim as optim
from collections import deque
from nnModel import PolicyNN
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train(num_episodes=5000, q_size=100, gamma=1.0, model=None):
    """

    Trains the neural net
    for every episode, it sees how far the pole can get in the simulation.
    Then it updates everything, and reruns until convergence
    """

    # maximum of 500 steps for this problem
    num_steps = 500
    reward_q = deque(maxlen=q_size)
    score_over_time = []
    loss_over_time = []
    for episode in range(num_episodes):
        state = env.reset()
        probs = []
        rewards = []
        for t in range(num_steps):
            action, prob = model.act(state)
            probs.append(prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break

        # from stackoverflow, use this equation instead
        # r1 + gamma * r2 + gamma ^ 2 * r3 + gamma ^ 3 * r4...
        R = sum([r * gamma ** idx for idx, r in enumerate(rewards)])
        loss = []
        for log_prob in probs:
            loss.append(-log_prob * R)
        loss = torch.cat(loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # keep accumulating only 100 most recent rewards
        reward_q.append(sum(rewards))
        if episode % 100 == 0:
            avg = sum(reward_q) / len(reward_q)
            score_over_time.append(avg)
            loss_over_time.append(loss.item())
            print(f"Turn {episode} results: {avg} loss: {loss.item()}")
    torch.save(model.state_dict(), "model.pth")
    return [score_over_time, loss]


def visualize_loss(score, loss):
    fig, [ax0, ax1] = plt.subplots(1, 2)
    ax0.plot(score)
    ax0.set_title("Average score over time")

    ax1.plot(loss)
    ax1.set_title("Loss over time")
    plt.show()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# use 30 so it is 6 -> 30 -> 3
model = PolicyNN(n_intermediate=30).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
env = gym.make("Acrobot-v1")
env.seed(0)
render = True
[model_score, model_loss] = train(gamma=1, model=model)
if render:
    visualize_loss(model_score, model_loss)
