import numpy as np
import gym
import pickle
env = gym.make('gym_robot_arm:robot-arm-v0')
min_theta = np.radians(0)
max_theta = np.radians(90)
q_table_size = [2,2]
q_table_segment_size = (max_theta - min_theta) / q_table_size


def convert_state(real_state):
    q_state = np.floor((real_state - min_theta) / q_table_segment_size).astype(int)
    q_state = np.clip(q_state, 0, np.array(q_table_size) - 1)  #
    return tuple(q_state)


training = True
if (training):
    q = np.random.uniform(size=(q_table_size + [env.action_space.n]))
else:
    f = open('gym_robot_arm:robot-arm-v0.pkl', 'rb')
    q = pickle.load(f)
    f.close()

episodes = 100
max_steps = 100
alpha = 0.9
gamma = 0.9


epsilon =200
epsilon_decay_rate = 2 / episodes
rng = np.random.default_rng()
rewards_per_episode = np.zeros(episodes)
done = True
action = env.action_space.sample()
max_reward = -999
max_action = []
observation = env.reset()
print("Target là ",observation[0],observation[1])
for a in range(episodes):
    theta = (observation[2], observation[3])
    state = convert_state(theta)
    ep_reward = 0
    action_list = []
    env.render()

    for b in range(max_steps):
        if training:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state])

            if not training:
                action = np.argmax(q[state])
        action_list.append(action)
        new_theta, reward, done, info = env.step(action)
        ep_reward += reward
        distance_error = info['distance_error']
        new_state = convert_state((new_theta[2], new_theta[3]))

        current_q=q[state+(action,)]

        q += alpha * ( reward + gamma * np.max(q[new_state]) - q[state][action])

        q[state+(action,)]=current_q
        state = new_state
        ep_reward += reward

        #print("Q Table là",q)
        if distance_error < epsilon:
            print("giá trị theta là ",theta)
            print("Đã chạm tới đích tại eps={},steps={}".format(a,b))
            print("giá trị distance: ",distance_error)
            done=True
            print ("giá trị theta={},action={}".format(new_theta,action))
        # if distance_error>epsilon:
        #     print("Chưa tới")
        if training:
            f = open('gym_robot_arm:robot-arm-v0.pkl', 'wb')
            pickle.dump(q, f)
            f.close()
