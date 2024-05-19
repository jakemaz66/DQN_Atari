import actor as a, critic as c, config as fig
from torch.optim import Adam
import torch
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
        """Take an action in the environement by sampling from policy"""
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
        self.log_probs.clear()
        self.states.clear()
        self.dones.clear()


    def sample_memory(self):
        """Collect mini-batches of samples, need to be randomly shuffled to avoid temporal correlation"""
        n_states = len(self.states)
        #Skip by batch size
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
        states_samples, action_samples, log_prob_samples, state_val_samples, reward_samples, dones_samples, batches = self.sample_memory()
        
        #Advantage Calculation for each step t -> A1, ..., AT
        advantage = np.zeros(len(reward_samples), dtype=np.float32)

        for t in range(len(reward_samples)-1):
            discount = 1
            #Tracking advantage value for each time step in trajectory
            a_t = 0
            for k in range(t, len(reward_samples)-1):
                #Discounted rewards plus the state at the next time-step k+1
                # r(t + 1) + V(s'), value of a terminal state is always 0
                a_t += discount*(reward_samples[k] + fig.GAMMA*state_val_samples[k+1]*\
                        (1-int(dones_samples[k])) - state_val_samples[k])
                #Incrementing discount 
                discount *= fig.GAMMA*fig.GAE_LAMBDA
            advantage[t] = a_t

        reward_samples = torch.tensor(reward_samples)
        advantage = torch.tensor(advantage)

        #Optimize surrogate L wrt θ, with K epochs and minibatch size M ≤ NT
        #Use mini-batches to repeatedly optimize
        for batch in batches:
            states_arr = torch.tensor(states_samples[batch]).squeeze(1)
            action_arr = torch.tensor(action_samples[batch])

            #The old log prob of actions we took during the trajectory
            old_probs = torch.tensor(log_prob_samples[batch])

            #Retrieving new policy from sampled states
            new_policy = self.actor(states_arr)

            ent = new_policy.entropy().mean()

            #New state values
            new_critic_value = self.critic(states_arr)
            new_probs = new_policy.log_prob(action_arr)

            #Ratio of new policy to old
            ratio = new_probs.exp() / old_probs.exp()
            weighted_ratio = ratio * advantage[batch]

            weighted_clipped_probs = torch.clamp(ratio, 1-fig.EPSILON, 1+fig.EPSILON) * advantage[batch]

            returns = advantage[batch] + state_val_samples[batch]
            actor_loss = -torch.min(weighted_ratio, weighted_clipped_probs).mean()

            critic_loss = (returns - new_critic_value).pow(2).mean()

            #total_loss = actor_loss - (ent * fig.ENTROPY_TERM) + 0.5*critic_loss
            total_loss = actor_loss  + 0.5*critic_loss

            #Perform updates
            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            self.critic_optimizer.step()
            self.actor_optimizer.step()




