import function_approximators, config
import torch
from torch.distributions import Categorical
import numpy as np

class Agent:

    def __init__(self, observation_size, action_size, learning_rate=3e-4):
        self.learning_rate = learning_rate
        self.function_approximator = function_approximators.FunctionApproximator(observation_size, action_size, hidden_size=256)
        self.returns = []
        self.actions = []
        self.state_vals = []
        self.log_probs = []
    
    def add_optm(self, optm):
        self.optm = optm

    def act(self, state):
        state = torch.tensor(state)
        state_val, policy = self.function_approximator(state)

        sampler = Categorical(policy)
        #Sampling an action from the policy
        action = sampler.sample()

        return state_val, policy, action.item()
    
    def add_reward(self, reward):
        self.returns.append(reward)
    
    def add_log_probs(self, log_prob):
        self.log_probs.append(log_prob)

    def add_state_vals(self, state_val):
        self.state_vals.append(state_val)

    def add_action(self, action):
        self.actions.append(action)

    def zero_episode(self):
        self.actions.clear()
        self.returns.clear()
        self.state_vals.clear()
        self.log_probs.clear()

    def learn(self, log_probs, value_next, done, entropy_term):
        
        # if done:
        #     value_next = 0

        log_probs = torch.stack(log_probs)


        #Computing discounted Q-values
        q_vals = np.zeros_like(self.state_vals)
        for t in reversed(range(len(self.returns))):
            value_next = self.returns[t] + config.GAMMA * value_next
            q_vals[t] = value_next

        q_vals = torch.FloatTensor(q_vals)
        state_vals = torch.FloatTensor(self.state_vals)

        advantage = q_vals - state_vals

        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()

        total_loss = actor_loss + critic_loss + 0.001 * entropy_term

        self.optm.zero_grad()
        total_loss.backward()
        self.optm.step()

        
