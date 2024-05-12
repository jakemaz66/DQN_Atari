import gym
import config, agent
import numpy as np
from torch import optim
import torch
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state, _ = env.reset()
    agent = agent.Agent(env.observation_space.shape[0], env.action_space.n, learning_rate=3e-4)
    ac_optimizer = optim.Adam(agent.function_approximator.parameters(), lr=config.LR)
    agent.add_optm(ac_optimizer)
    

    entropy_term = 0
    all_rewards = []
    all_lengths = []
    average_lengths = []


    for i in range(config.NUMEPISODES):
        #After each update clear out the episode rewards and actions
        agent.zero_episode()
        count = 0
        state, info = env.reset()

        for j in range(config.NUMSTEPS):
            count += 1
            #My self.state_vals are blowing up
            state_val, policy, action = agent.act(state)
            state_val = state_val.detach().numpy()[0]
            log_prob = torch.log(policy.squeeze(0)[action])
            entropy = -np.sum(np.mean(policy.detach().numpy()) * np.log(policy.detach().numpy()))

            #Why is terminated always true
            obervation, reward, terminated, truncated, info = env.step(action)

            agent.add_reward(reward)
            agent.add_action(action)
            agent.add_state_vals(state_val)
            agent.add_log_probs(log_prob)
            entropy_term += entropy

            state = obervation

            if terminated or truncated or count == (config.NUMSTEPS - 1):

                all_rewards.append(np.sum(agent.returns))
                all_lengths.append(count)
                average_lengths.append(np.mean(all_lengths[-10:]))
                value_next, _ = agent.function_approximator(obervation)
                value_next = value_next.detach().numpy()[0]
                if i % 10 == 0:                    
                    print("episode: {}, reward: {}, total length: {}, average length: {} \n".format(i, np.sum(agent.returns), count, average_lengths[-1]))     

                break

        agent.learn(agent.log_probs, value_next, True, entropy_term)

    smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(all_rewards)
    plt.plot(smoothed_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.plot(all_lengths)
    plt.plot(average_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.show()

        

        



