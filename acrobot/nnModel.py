import torch.nn
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

"""

NN Model we will use for the acrobot
NFeed forward eural net with 6 input layers,
one hidden layer of 12 units and relu activation,
and output layer of 3 units and softmax

"""


class PolicyNN(nn.Module):
    def __init__(self):
        super(PolicyNN, self).__init__()
        # have 2 layers, our 6 feed into 12 into 3
        self.fc1 = nn.Linear(6, 12)
        # applies a fully connected linear transformation to our data
        self.fc2 = nn.Linear(12, 3)

    def forward(self, x):
        # just feeds forward
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        # just makes a tensor object, (4,)
        state = torch.from_numpy(state).float().unsqueeze(0)
        # gets a tensor(2,) about probability of 1 or probability of 1
        probs = self.forward(state)
        # Creates a categorical distribution
        m = Categorical(probs)
        # gets the highest prob
        action = m.sample()
        return action.item() - 1, m.log_prob(action)
