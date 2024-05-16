import gym
import agent as a
import config as c
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio


if __name__ == '__main__':
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    obs_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    agent = a.Agent(learning_rate=c.LR, obs_dims=obs_space, action_dims=action_space)
    reward_dict = {}
    entropy_value = 0
    frames = []

    for episode in range(c.EPISODES):
        state, _ = env.reset()
        ep_reward = 0

        for step in range(c.NUMSTEPS):
            if episode % 3400 == 0:
                frame = env.render()
                frames.append(frame)

            state_value, action, log_prob, policy = agent.act(state)

            observation, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            entropy = -np.sum(np.mean(policy.detach().numpy()) * np.log(policy.detach().numpy()))
            entropy_value += entropy

            agent.collect(reward, state_value, log_prob)

            state = observation

            #If the episode is over or number of steps reached, update network
            if (terminated or truncated or step == c.NUMSTEPS-1):
                agent.learn(observation, entropy_value)
                agent.clear_collect()
                print(f"Episode {episode} complete with reward of {ep_reward}")
                reward_dict[str(episode)] = ep_reward
                break

        


    rewards = pd.Series(reward_dict)
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.show()
    plt.savefig('A2C_Scratch.png')

    imageio.mimwrite('lunar_lander.gif', frames, fps=60)




        