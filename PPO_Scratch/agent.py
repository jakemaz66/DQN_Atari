import actor as a, critic as c, config as fig
from torch.optim import Adam
import torch
from torch.distributions import Categorical
import numpy as np
from stable_baselines3 import PPO


class Agent:

    def __init__(self, obervation_size, action_size, batch_size):
        self.actor = a.Actor(obervation_size, action_size)
        self.critic = c.Critic(obervation_size, action_size)

        self.actor_optimizer = Adam(params=self.actor.parameters(), lr=fig.LR)
        self.critic_optimizer = Adam(params=self.critic.parameters(), lr=fig.LR)

        self.batch_size = batch_size

        #We need to keep track of trajectories, this is the agent's 'memory'
        self.actions = []
        self.rewards = []
        self.state_vals = []
        self.log_probs = []
        self.states = []
        self.dones = []

        self.old_policy = None
        

    def act(self, observation):
        observation = torch.tensor(observation)

        policy = self.actor(observation)
        state_value = self.critic(observation)
        action = policy.sample()
        log_prob = policy.log_prob(action)

        return action.item(), log_prob.detach().numpy(), state_value.detach().numpy()
    

    def collect(self, reward, action, state_val, log_prob, states, dones):
        self.actions.append(action)
        self.rewards.append(reward)
        self.state_vals.append(state_val)
        self.log_probs.append(log_prob)
        self.states.append(states)
        self.dones.append(dones)

    def clear(self):
        self.actions.clear()
        self.rewards.clear()
        self.state_vals.clear()

    def sample_memory(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.log_probs),\
                np.array(self.state_vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def learn(self):
        #Sampling minibatches from memory of trajectory
        states_samples, action_samples, log_prob_samples, state_val_samples, reward_samples, dones_samples, batch = self.sample_memory()
        
        #Advantage Calculation for each step t -> A1, ..., AT
        advantage = np.zeros(len(reward_samples), dtype=np.float32)

        for t in range(len(reward_samples)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_samples)-1):
                #Rewards plus the state at the next time-step k+1
                a_t += discount*(reward_samples[k] + fig.GAMMA*state_val_samples[k+1]*\
                        (1-int(dones_samples[k])) - state_val_samples[k])
                discount *= fig.GAMMA*fig.GAE_LAMBDA
            advantage[t] = a_t

        rewards = torch.tensor(reward_samples)
        advantage = torch.tensor(advantage)

        #Optimize surrogate L wrt θ, with K epochs and minibatch size M ≤ NT
        #Use mini-batches to repeatedly optimize
        for batch in batch:
            states_samples = torch.tensor(states_samples[batch]).squeeze(1)
            action_samples = torch.tensor(action_samples[batch])
            log_prob_samples = torch.tensor(log_prob_samples[batch])

            #The old log prob of actions we took during the trajectory
            old_probs = log_prob_samples
            #Retrieving new policy from sampled states
            new_policy = self.actor(states_samples)
            new_critic_value = self.critic(states_samples)
            new_probs = new_policy.log_prob(action_samples)

            ratio = new_probs.exp() / old_probs.exp()
            weighted_ratio = ratio * advantage[batch]

            weighted_clipped_probs = torch.clamp(ratio, 1-fig.EPSILON, 1+fig.EPSILON) * advantage[batch]

            returns = advantage[batch] + state_val_samples[batch]
            actor_loss = -torch.min(weighted_ratio, weighted_clipped_probs).mean()

            critic_loss = (returns + state_val_samples[batch] - new_critic_value).pow(2).mean()

            total_loss = actor_loss + 0.5*critic_loss


            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            self.critic_optimizer.step()
            self.actor_optimizer.step()




