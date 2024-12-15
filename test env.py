import numpy as np
import pickle
import gym

def run_robot_arm(episodes, is_training=True, render=False):
    env = gym.make('gym_robot_arm:robot-arm-v0')
    min_theta=np.radians(0)
    max_theta=np.radians(90)
    max_length=200
    pos_space = np.linspace(-max_length,max_length, 20)
    theta_space = np.linspace(min_theta, max_theta, 20)
    if is_training:
        q = np.zeros((len(pos_space), len(pos_space), len(theta_space), len(theta_space), env.action_space.n))
    else:
        with open('robot_arm_q_table.pkl', 'rb') as f:
            q = pickle.load(f)
    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1
    epsilon_decay_rate = 1 / episodes
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)
    for i in range(episodes):
        state,info = env.reset()
        initial_tip_position = info['initial_tip_position']
        state = np.array(state, dtype=np.float32)
        theta1, theta2 = state[2],state[3]
        a,b=initial_tip_position
        posx=np.digitize(a,pos_space)-1
        posy=np.digitize(b,pos_space)-1
        state_t1 = np.digitize(theta1, theta_space) - 1
        state_t2 = np.digitize(theta2, theta_space) - 1
        terminated = False
        total_reward = 0

        while not terminated and total_reward>-10:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_t1, state_t2, :])
                action=np.clip(action,0,6)
            new_state, reward, terminated, info= env.step(action)
            theta1, theta2 = new_state[0],new_state[1]
            current_position = info['current_position']
            x, y = current_position
            new_posx=np.digitize(x,pos_space)-1
            new_posy=np.digitize(y,pos_space)-1
            new_state_t1 = np.digitize(theta1, theta_space) - 1
            new_state_t2 = np.digitize(theta2, theta_space) - 1
            if is_training:
                q[ state_t1, state_t2,posx,posy, action] = (
                    q[state_t1, state_t2,posx,posy,action]
                    + learning_rate_a
                    * (
                        reward
                        + discount_factor_g * np.max(q[new_state_t1, new_state_t2,new_posx,new_posy, :])
                        - q[state_t1, state_t2,posx,posy, action]
                    )
                )
            state_t1, state_t2 =  new_state_t1, new_state_t2
            posx,posy=new_posx,new_posy
            total_reward += reward
            if render:
                env.render()
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        rewards_per_episode[i] = total_reward
    if is_training:
        with open('robot_arm_q_table.pkl', 'wb') as f:
            pickle.dump(q, f)
    env.close()
if __name__ == "__main__":
    run_robot_arm(500, is_training=True, render=False)
    run_robot_arm(100, is_training=False, render=True)
