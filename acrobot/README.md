# Acrobot

The acrobot system includes two joints and two links, where the joint between the two links is actuated. Initially, the links are hanging downwards, and the goal is to swing the end of the lower link up to a given height.

## Progression

+ Ultimately, we are trying to reach an average return of 487.5 over 100 trials.
    + This is 97.5%.
+ The trial can go a maximum of 500 steps.
+ The options are -1 (left), 0 (stay still), or 1 (right)



### Step 1
Random
```python
import random

def select_action_(state):
    rand = random.random()
    if rand < 0.33:
        return 1
    elif rand < 0.66:
        return 0
    return -1
```
+ Average return: 0

[comment]: <> (### Step 1)

[comment]: <> (Started first with a version that noted "if the pole angle is going left, make it go right")

[comment]: <> (```python)

[comment]: <> (import gym)

[comment]: <> (env = gym.make&#40;'CartPole-v1'&#41;)

[comment]: <> (state = env.reset&#40;&#41;)

[comment]: <> (while True:)

[comment]: <> (    action = 0 if state[2] <= 0 else 1)

[comment]: <> (    state, _, done, _ = env.step&#40;action&#41;)

[comment]: <> (    env.render&#40;&#41;)

[comment]: <> (    if done:)

[comment]: <> (        break)

[comment]: <> (env.close&#40;&#41;)

[comment]: <> (```)

[comment]: <> (+ Average return: 42)

[comment]: <> (+ Issues: does not account for the velocity of the cart, just what direction it is going in.)

[comment]: <> (### Step 2: include velocity)

[comment]: <> (Now instead of just relying on if the pole is going left, we want to account for "how much" it is going left.)

[comment]: <> (Instead of just using pole angle, we also use pole angular velocity.)

[comment]: <> (#### Solution 1)

[comment]: <> (```python)

[comment]: <> (import gym)

[comment]: <> (env = gym.make&#40;'CartPole-v1'&#41;)

[comment]: <> (state = env.reset&#40;&#41;)

[comment]: <> (while True:)

[comment]: <> (    action = 0 if state[2]*state[3] <= 0 else 1)

[comment]: <> (    state, _, done, _ = env.step&#40;action&#41;)

[comment]: <> (    env.render&#40;&#41;)

[comment]: <> (    if done:)

[comment]: <> (        break)

[comment]: <> (env.close&#40;&#41;)

[comment]: <> (```)

[comment]: <> (+ Multiplying the angle by the velocity)

[comment]: <> (+ Average return: 127)

[comment]: <> (#### Solution 2)

[comment]: <> (```python)

[comment]: <> (import gym)

[comment]: <> (env = gym.make&#40;'CartPole-v1'&#41;)

[comment]: <> (state = env.reset&#40;&#41;)

[comment]: <> (while True:)

[comment]: <> (    action = 0 if state[2] + state[3] <= 0 else 1)

[comment]: <> (    state, _, done, _ = env.step&#40;action&#41;)

[comment]: <> (    env.render&#40;&#41;)

[comment]: <> (    if done:)

[comment]: <> (        break)

[comment]: <> (env.close&#40;&#41;)

[comment]: <> (```)

[comment]: <> (+ Average return: 199)

[comment]: <> (+ Technically this fully passes the reinforcement learning task, but still doesn't use reinforcement learning)

[comment]: <> (### Step 3: Use reinforcement learning)

[comment]: <> (Since the above worked so well, we will now try and predict how it works.)

[comment]: <> (+ Essentially, our action is determined from the state &#40;arr len4&#41; and returns either 0 or 1.)

[comment]: <> (+ So we will make a neural net to determine, given the state, the probability of 1 or 0)


[comment]: <> (+ Average return: 199)