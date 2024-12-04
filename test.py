import gym
import time
env = gym.make('gym_robot_arm:robot-arm-v3')
for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        time.sleep(1)
        print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()
