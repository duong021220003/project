import numpy as np
import gym

env = gym.make('gym_robot_arm:robot-arm-v0')
min_theta = np.radians(0)
max_theta = np.radians(90)
q_table_size = [10, 10]
q_table_segment_size = (max_theta - min_theta) / q_table_size


def convert_state(real_state):
    q_state = np.floor((real_state - min_theta) / q_table_segment_size).astype(int)
    q_state = np.clip(q_state, 0, np.array(q_table_size) - 1)  # Ensure it's within bounds
    return tuple(q_state)


q = np.random.uniform(size=(q_table_size + [env.action_space.n]))
episodes = 1000
max_steps = 200
alpha = 0.9
gamma = 0.9
training = True

show = 10
epsilon = 1
epsilon_decay_rate = 2 / episodes
rng = np.random.default_rng()
rewards_per_episode = np.zeros(episodes)
done = True
action = env.action_space.sample()
max_reward = -999
max_action = []
observation = env.reset()
for i in range(episodes):
    theta = [observation[2], observation[3]]
    state = convert_state(theta)
    ep_reward = 0
    action_list = []
    env.render()
    for _ in range(max_steps):


        new_theta, reward, done, info = env.step(action)
        ep_reward += reward
        distance_error = info['distance_error']


        if done:
            if distance_error < 0.1:
                print("Đã chạm tới đích,")
        else:
            if training:
                if rng.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q[state])

                if not training:
                    action = np.argmax(q[state])
            action_list.append(action)
            new_state = convert_state((new_theta[2], new_theta[3]))
            q[state][action] += alpha * (
                    reward + gamma * np.max(q[new_state]) - q[state][action]
            )
            state = new_state

