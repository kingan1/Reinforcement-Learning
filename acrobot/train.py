"""

Training the acrobot model

"""

import gym

# Neural net
import torch
import torch.optim as optim
from collections import deque
from nnModel import PolicyNN


def train(num_episodes=5000, gamma=1):
    """

    Trains the neural net
    for every episode, it sees how far the pole can get in the simulation.
    Then it updates everything, and reruns until convergence
    """

    model = PolicyNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_steps = 500
    reward_100 = deque(maxlen=100)
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
        loss = sum(loss)
        reward_100.append(sum(rewards))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            avg = sum(reward_100) / len(reward_100)
            print(f"Turn {episode} results: {avg}")
    torch.save(model.state_dict(), "model.pth")


env = gym.make("Acrobot-v1")
env.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train()