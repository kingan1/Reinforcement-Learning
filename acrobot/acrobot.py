"""

Training the acrobot model

"""

import gym
# Neural net
import torch
import torch.optim as optim

from nnModel import PolicyNN


def train(num_episodes=1000):
    """

    Trains the neural net
    for every episode, it sees how far the pole can get in the simulation.
    Then it updates everything, and reruns until convergence
    """

    model = PolicyNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_steps = 500
    ts = []
    for episode in range(num_episodes):
        state = env.reset()
        probs = []
        rewards = []
        for t in range(1, num_steps + 1):
            action, prob = model.act(state)
            probs.append(prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        # calculates loss function
        # sums all of the rewards (either -1 or 0)
        R = rewards.sum()

        policy_loss = []
        # for each probability, appends the prob * R
        for log_prob in prob:
            policy_loss.append(-log_prob * R)
        loss = policy_loss.sum()

        # sets all gradients to 0
        optimizer.zero_grad()
        # accumulates the gradients
        loss.backward()
        # paramter updated based on current parameters
        optimizer.step()
        # curr_iter
        ts.append(t)
    torch.save(model.state_dict(), "model.pth")


env = gym.make('Acrobot-v1')
env.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train()
