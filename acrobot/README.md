# Acrobot

The acrobot system includes two joints and two links, where the joint between the two links is actuated. Initially, the links are hanging downwards, and the goal is to swing the end of the lower link up to a given height.

## Progression

+ Ultimately, we are trying to reach an average return of 487.5 over 100 trials.
    + This is 97.5%.
+ The trial can go a maximum of 500 steps.
+ The options are -1 (left), 0 (stay still), or 1 (right)
+ state is represented as  [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].


### Step 0
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
+ Percentage worked: 0

### Step 1

Tried to use some mathematics, no soundproof logic behind it
```python
def select_action_(state):
    top_joint = state[2:4]
    # cos 1 = pointing down
    # when the angular velocity slows/stops, go other way

    s3 = state[4]
    s4 = state[5]
    if s3 + s4 < 0.1:
        # if it's slowing down, go the other way
        if top_joint[1] < 0:
            return -1
        return 1
    else:
        if top_joint[1] < 0:
            return -1
        elif top_joint[1] == 0:
            return 0
        return 1

```

+ Percentage worked: 60%

### Step 3: Use reinforcement learning

We will now use reinforcement learning. This was harder than cart_pole becase

1. 3 outputs, not 2
  + had to also customize instead of a range of [0-2], have it [0-1]
2. Needed more layers
  + Before I was able to get away with one layer but now I had to add a hidden input layer
3. Needed a different loss
  + Before it was just maximize the amount of turns still active, now I had to minimize the number of turns active.