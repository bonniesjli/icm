[//]: # (Image References)

[image1]: https://github.com/bonniesjli/icm/blob/master/pictures/pyramid.png "pyramid"
[image2]: https://github.com/bonniesjli/icm/blob/master/pictures/pyramid_learning.png "pyramid_agent"
[image3]: https://github.com/bonniesjli/icm/blob/master/pictures/push.png "push"
[image4]: https://github.com/bonniesjli/icm/blob/master/pictures/push_learning.png "push_agent"

# PPO+ICM

This is an implementation of intrinsic curiosity module (ICML 2017).

### Tests
#### Pyramid env, Unity ML
Agent Reward Function (independent): <br>
    +2 For moving to golden brick 
    - 0.001 per step<br>
![pyramid][image1]
![pyramid_agent][image2]

#### PushBlock env, Unity ML
Agent Reward Function: 
+5.0 if the block touches the goal<br>
-0.0025 for every step.<br>
![push][image3]
![push_agent][image4]

### ICM Module Usage
Located in `icm.py`
* initialize module within ppo agent <br>
`class Agent():`<br>
`  def __init__():`<br>
`    self.icm = ICM(state_size, action_size)`<br>
* compute intrinsic reward when interacting with environment <br>
`intrinsic_reward = agent.icm.compute_intrinsic_reward(states, next_states, actions)`<br>
* train ICM when training PPO <br>
`self.icm.train(state_samples, next_state_samples, action_samples)`<br>