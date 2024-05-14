import config, function_approximators, memory_replay
import gymnasium as gym
import torch
from torch import nn
import numpy as np
import torch.optim as optim
from collections import deque
from wrapper import make_env
import matplotlib.pyplot as plt
import math


def update_params():
    """This function updates the parameters of the policy network"""

    #If the agent has not accumulated enough experience yet
    if len(replay_memory)< config.BATCHSIZE:
        return
    
    #Sampling from the replay memory, samples is a batch of named tuples
    #Weights are from prioritized replay importance sampling
    samples = replay_memory.sample(config.BATCHSIZE)

    #Gathering all the states, actions, and rewards from sample
    batch = memory_replay.EnvStep(*zip(*samples))

    #Creating new tensors of concatenated values along the dimensions of the named tuple
    state_batch = torch.stack(batch.state, dim=0).to(torch.float32)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    #Gets all Q-values for the states in the sampled episodes
    #Then, retrieve q values of the chosen actions
    q_value_function = policy_network(state_batch).gather(1, action_batch.unsqueeze(0))

    #Getting the state value function at the next time step
    next_step_value_function = torch.zeros(config.BATCHSIZE)
    #Getting all states that are not terminal from the memory (we don't want to learn on terminal states)
    non_final_mask =  torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
    
    #Get all the non_final states states for time step t + 1
    nf_states = [s for s in batch.next_state if s is not None]
    non_final_states =  torch.stack(nf_states, dim=0).to(torch.float32)
    with torch.no_grad():
        #Getting the predicted Q values from the target network, taking the max Q-value
        next_step_value_function[non_final_mask] = target_network(non_final_states).max(1).values

    #Now, compute the state-action Q function (with discount factor)
    next_q_value_function = (next_step_value_function * config.GAMMA) + reward_batch

    #The loss is the initial q value estimate subtracted by the estimate after one time step (after we gain more information)
    #q_value_function is the Q values of the actions we did take, next_q_val is the immediate reward + the max Q val of the next state
    criterion = nn.SmoothL1Loss()
    loss = criterion(q_value_function.squeeze(0), next_q_value_function)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_network.parameters(), 100)
    optimizer.step()


#Main training loop
if __name__ == '__main__':


    """Pong Actions:
        0: Noop
        1: Fire
        2: Right
        3: Left
        4: Right Fire
        5: Left Fire

    """
    #Setting up the gym environment
    env = make_env("PongNoFrameskip-v4")

    #observation_space is a Box() class of coordinate positions, access dimensions with _shape
    action_space = env.action_space.n

    #Defining Q function approximators with inputs of 4 frames by size 84x84
    policy_network = function_approximators.AtariFunctionApproximator((4,84,84), action_space)
    target_network = function_approximators.AtariFunctionApproximator((4,84,84), action_space)

    #Setting the parameters of the target network equal to the policy network
    target_network.load_state_dict(policy_network.state_dict())

    optimizer = optim.AdamW(policy_network.parameters(), lr=config.LR, amsgrad=True)

    #Initializing optimizer and prioritized replay memory
    replay_memory = memory_replay.ReplayMemory(capacity=100_000)
    epsilon = 1

    #State is an image
    state, info = env.reset()
    state = torch.tensor(state)

    n_episode = 0
    episode_reward = 0

    #Running avwrage of reward for metric evaluation
    last_100_avg = [-21]
    score = deque(maxlen=100)

    for i in range(config.NUMFRAMES):
            with torch.no_grad():
                if state.shape != torch.Size([4,84,84]):
                    state = torch.tensor(state).permute(2,0, 1)

                new_epslion, action = policy_network.act(state, epsilon, env)
                eps_threshold = config.EPS_END + (config.EPS_START - config.EPS_END) * math.exp(-1. * i / config.EPS_DECAY)
                epsilon = new_epslion

            observation, reward, terminated, truncated, info = env.step(action.item())

            observation = torch.tensor(observation)
            
            reward = torch.tensor([reward])
            
            replay_memory.push(state, action, observation, reward)

            done = terminated or truncated
            episode_reward += reward    

            #Replace the current state with the next observation from the environment
            state = observation

            #update the policy network parameters
            update_params()


            #update the parameters of the target network every 10,000 frames
            if i % 10_000 == 0:
                target_sd = target_network.state_dict()
                policy_sd = policy_network.state_dict()

                target_network.load_state_dict(policy_sd)

            #Saving model every 100,000 frames
            if i % 100_000 == 0:
                weights_path = f'DQN_Atari/saved_models/model_{i}.pth'

                # Save the model's state_dict (weights and biases)
                torch.save(target_network.state_dict(), weights_path)
                
            #If episode is over, break out of loop and iterate episode
            if done:

                state, info = env.reset()
                state = torch.tensor(state)
                n_episode += 1
                score.append(episode_reward.item())

                print(f'Episode {n_episode} is complete')
                print(f'Reward at this episode: {episode_reward}')
                episode_reward = 0

                if n_episode % 10_000 == 0:
                    last_100_avg.append(sum(score)/len(score))
                    plt.plot(np.arange(0,n_episode+1,10),last_100_avg)
                    plt.show()

