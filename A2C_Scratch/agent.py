import neural_net as n
from torch.optim import RMSprop
import numpy as np
import torch
import config as c
from torch.distributions import Categorical


class Agent:

    def __init__(self, learning_rate, obs_dims, action_dims):
        self.neural_net = n.NeuralNet(obs_dims, action_dims)
        self.optimizer = RMSprop(params= self.neural_net.parameters(), lr=learning_rate)
        #Use numpy arrays to make faster
        self.state_value = []
        self.rewards = []
        self.log_probs = []

    def collect(self, reward, state_value, log_prob):
        self.state_value.append(state_value)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def act(self, observation):
        observation = torch.tensor(observation)
        state_value, policy = self.neural_net(observation)
        policy_act = Categorical(policy)

        action = policy_act.sample()
        log_prob = policy_act.log_prob(action)

        return state_value.detach().numpy()[0], action.detach().numpy(), log_prob, policy
    
    def new_state_value(self, obs):
        obs = torch.tensor(obs)
        new_state_value, _ = self.neural_net(obs)
        return new_state_value.detach().numpy()[0]
        

    def learn(self, observation, entropy):
        observation = torch.tensor(observation)
        new_state_value, _ = self.neural_net(observation)
        new_state = new_state_value.detach().numpy()[0]

        new_state_values = np.zeros_like(self.state_value)
        for t in reversed(range(len(self.rewards))):
            new_state = self.rewards[t] + c.GAMMA * new_state
            new_state_values[t] = new_state

        state_value = torch.tensor(self.state_value)
        log_probs = torch.stack(self.log_probs)

        #Critic Loss -> (Advantage)^2
        rewards = torch.tensor([self.rewards])
        advantage = (rewards + c.GAMMA * new_state_values) - state_value
        critic_loss = advantage.pow(2).mean()

        #Actor loss
        actor_loss = -(log_probs * advantage).mean()

        total_loss = (critic_loss + actor_loss) + (c.BETA * entropy)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def clear_collect(self):
        self.state_value.clear()
        self.rewards.clear()
        self.log_probs.clear()


