import numpy as np
import gym
import pickle


env = gym.make('gym_robot_arm:robot-arm-v0',)


min_theta = np.radians(0)
max_theta = np.radians(90)
q_table_size = [2,2]
q_table_segment_size = (max_theta - min_theta) / q_table_size

def convert_state(real_state):
    q_state = np.floor((real_state - min_theta) / q_table_segment_size).astype(int)
    q_state = np.clip(q_state, 0, np.array(q_table_size) - 1)  #
    return tuple(q_state)

training=True
episodes=10
if (training):
    q = np.zeros((2,2, env.action_space.n))
else:
    f = open('gym_robot_arm:robot-arm-v0.pkl', 'rb')
    q = pickle.load(f)
    f.close()

alpha=0.9
gamma=0.9
epsilon = 1  # 1 = 100% random actions
epsilon_decay_rate = 2 / episodes  # epsilon decay rate
rng = np.random.default_rng()
rewards_per_episode = np.zeros(episodes)


for i in range(episodes):
    observation=env.reset()
    theta=(observation[2],observation[3])
    state=convert_state(theta)
    terminated=False
    rewards=0
    print(observation)
    



    
    if training and rng.random()<epsilon:
        action=env.action_space.sample()

    else:
        action = np.argmax(q[state, :])

    new_state,reward,terminated,_=env.step(action)
    new_state=convert_state(new_state)

    if training:
        q[state,action]=q[state,action]+alpha*(reward+ gamma*np.max(q[new_state,:])-q[state,action]
                                                )
    theta=new_state
    state=convert_state(new_state)
    rewards+=reward

epsilon = max(epsilon - epsilon_decay_rate, 0)

rewards_per_episode[i] = rewards







