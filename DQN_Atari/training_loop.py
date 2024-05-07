import gym
import torch
from torch import nn
import random
from DQN_Atari import config, function_approximators, memory_replay
import torch.optim as optim
from itertools import count

#Setting up the gym environment
env = gym.make('CartPole-v1')
state, info = env.reset()
observation_space = len(env.observation_space)
action_space = len(env.action_space)

#Defining Q function approximators
policy_network = function_approximators.FunctionApproximator(observation_space, action_space)
target_network = function_approximators.FunctionApproximator(observation_space, action_space)
#Setting the parameters of the target network equal to the policy network
target_network.load_state_dict(policy_network.state_dict())

optm = optim.Adam(policy_network.parameters(), lr=config.LR)
replay_memory = memory_replay.MemoryReplay(max_len=10_000)


def take_action(observation):
    """This function takes one action (E-Greedy selection of Q values)"""
    epsilon = config.EPSILON

    #Sampling a random float between 0 and 1
    explore = round(random.uniform([0,1], 2))

    #Because we are not updating parameters
    with torch.no_grad():

        if explore > epsilon:

            #Take the action with the highest Q value (max(1)) and return a tensor with it
            max_q = policy_network(observation).max(1).indices.view(-1, 1)
            return max_q
        
        else:
            #Randomly sample an action (exploring)
            random_q = random.sample(env.action_space, 1)
            return torch.tensor(random_q)



def update_params():
    """This function updates the parameters of the policy network"""

    #If the agent has not accumulated enough experience yet
    if len(replay_memory) < config.BATCHSIZE:
        return
    
    #Sampling from the replay memory, samples is a batch of named tuples
    samples = replay_memory.sample(config.BATCHSIZE)

    #Gathering all the states, actions, and rewards from sample
    batch = memory_replay.EnvStep(*zip(*samples))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))
    #get all the states for time step t + 1
    non_final_states = torch.cat(lambda s: s for s in batch.next_state if s is not None)

    #Creating new tensors of concatenated values along the dimensions of the named tuple
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    #Gets all policies for the states in the sampled episodes, then gets all the actions 
    #This gets us the actions that would have been taken under our current policy
    q_value_function = policy_network(state_batch).gather(1, action_batch)

    #Getting the state value function at the next time step
    next_step_value_function = torch.zeros(config.BATCHSIZE)
    with torch.no_grad():
        #Set all future values that are not final to
        next_step_value_function[non_final_mask] = target_network(non_final_states).max(1).values

    #Now, compute the state-action Q function (with discount factor)
    next_q_value_function = (next_step_value_function * config.GAMMA) + reward_batch

    #Computing the loss function
    loss_fn = nn.SmoothL1Loss()
    #The loss is the initial q value estimate subtracted by the estimate after one time step (after we gain more information)
    loss = loss_fn(q_value_function, next_q_value_function.unsqueeze(1))

    optm.zero_grad()
    loss.backward()

    #We clip the gradient in order to protect against exploding gradients
    torch.nn.utils.clip_grad_value_(policy_network.parameters(), 100)
    optm.step()



#Main training loop
if __name__ == '__main__':

    for i in range(config.NUMEPISODES):
        state, info = env.reset()
        state = torch.tensor(state).unsqueeze(0)

        #For as many time steps as it takes until the episode terminates
        for t in count():
            action = take_action(state)

            observation, reward, terminated, truncated, info = env.step(action.item())
            reward = torch.tensor(reward)
            done = terminated or truncated

            replay_memory.add_episode(state, action, observation, reward)

            #Replace the current state with the next observation from the environment
            state = torch.tensor(observation).unqueeze(0)

            #update the policy network parameters
            update_params()

            #update the parameters of the target network
            target_sd = target_network.state_dict()
            policy_sd = policy_network.state_dict()

            for key in policy_sd:
                target_sd[key] = (policy_sd[key] * config.SOFTUPDATE) + target_sd[key] * (1-config.SOFTUPDATE)

            target_network.load_state_dict(target_sd)

            #If episode is over, break out of loop and iterate episode
            if done:
                break
