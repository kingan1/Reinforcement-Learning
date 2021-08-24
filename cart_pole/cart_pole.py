# First project using reinforcement learning

import gym

# Neural net
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNN(nn.Module):
    def __init__(self):
        super(PolicyNN, self).__init__()
        # applies a fully connected linear transformation to our data
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        # just feeds forward
        x = self.fc(x)
        # generalization of logistic, takes in parameters and returns a probability
        # wikipedia:  last activation function of a neural network to normalize
        #     the output of a network to a probability distribution over predicted output classes,
        #     based on Luce's choice axiom.
        return F.softmax(x, dim=1)


# used during trainig
def select_action_from_policy(model, state):
    # just makes a tensor object, (4,)
    state = torch.from_numpy(state).float().unsqueeze(0)
    # gets a tensor(2,) about probability of 1 or probability of 1
    probs = model(state)
    # Creates a categorical distribution
    m = Categorical(probs)
    # gets the highest prob
    action = m.sample()
    return action.item(), m.log_prob(action)


def select_action_from_policy_best(model, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = model(state)
    if probs[0][0] > probs[0][1]:
        return 0
    else:
        return 1


env = gym.make('CartPole-v1')
render = False


def train_simple(num_episodes=10000):
    """

    Trains the neural net
    for every episode, it sees how far the pole can get in the simulation.
    Then it updates everything, and reruns until convergence
    """
    num_steps = 200
    ts = []
    for episode in range(num_episodes):
        state = env.reset()
        probs = []
        for t in range(1, num_steps + 1):
            action, prob = select_action_from_policy(model, state)
            probs.append(prob)
            state, _, done, _ = env.step(action)
            if done:
                break
        loss = 0
        # calculates loss function
        for i, prob in enumerate(probs):
            loss += -1 * (t - i) * prob
        # sets all gradients to 0
        optimizer.zero_grad()
        # accumulates the gradients
        loss.backward()
        # paramter updated based on current parameters
        optimizer.step()
        # curr_iter
        ts.append(t)
        # check stopping condition:
        if len(ts) > 10 and sum(ts[-10:]) / 10.0 >= num_steps * 0.95:
            print('Converged')
            return


def select_action_from_policy_test(model, state):
    """
    Used after the model is trained.
    Gets the probability given the state, and returns either 0/1
    """
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = model(state)
    return 0 if probs[0][0] > probs[0][1] else 1
    # if probs[0][0] > probs[0][1]:
    #     return 0
    # else:
    #     return 1


def test(total_sim=100, steps=200):
    # basic version, if it is going left, go right, etc
    sum_return = 0
    for _ in range(total_sim):
        # run this simulation 100 times
        state = env.reset()
        # keep track of what stage we are on
        for i in range(steps):
            res = select_action_from_policy_test(model, state)
            action = res
            state, reward, done, _ = env.step(action)
            if render:
                env.render()
            if done:
                break
        # Keep track of the return
        sum_return += i
    print(f'Average return: {sum_return / total_sim}')
    if render:
        env.close()


model = PolicyNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
train_simple()
test()
