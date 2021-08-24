# CartPole

Basic reinforcement learning project to balance a pole on top of a cart.

Designed using Python Machine Learning Second Edition and [bytepawn](https://bytepawn.com/solving-the-cartpole-reinforcement-learning-problem-with-pytorch.html)

## Progression

Ultimately, we are trying to reach an average return of 195.0 over 100 trials.
The cart can go a maximum of 200 steps.

### Step 1
Started first with a version that noted "if the pole angle is going left, make it go right"
```python
import gym
env = gym.make('CartPole-v1')
state = env.reset()
while True:
    action = 0 if state[2] <= 0 else 1
    state, _, done, _ = env.step(action)
    env.render()
    if done:
        break
env.close()
```
+ Average return: 7.71
+ Issues: does not account for the velocity of the cart, just what direction it is going in.

### Step 2: include velocity

Now instead of just relying on if the pole is going left, we want to account for "how much" it is going left.
Instead of just using pole angle, we also use pole angular velocity.