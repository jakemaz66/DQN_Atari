import config, function_approximators, memory_replay
import gym
import torch
from torch import nn
import numpy as np
import torch.optim as optim
from collections import deque
from preprocess_frames import preprocess_frames, stack_frames
import matplotlib.pyplot as plt


def initialize_game(env_name, optimizer, memory_replay_length):
    """This function initializes the gym environment, the function approximation networks, the optimizer,
       and the replay memory from which to sample transitions
    """
    #Setting up the gym environment
    env = gym.make(env_name)

    #observation_space is a Box() class of coordinate positions, access dimensions with _shape
    action_space = env.action_space.n

    #Defining Q function approximators with inputs of 4 frames by size 84x84
    policy_network = function_approximators.AtariFunctionApproximator((4,84,84), action_space)
    target_network = function_approximators.AtariFunctionApproximator((4,84,84), action_space)

    #Setting the parameters of the target network equal to the policy network
    target_network.load_state_dict(policy_network.state_dict())

    #Initializing optimizer and prioritized replay memory
    optm = optimizer(policy_network.parameters(), lr=config.LR, amsgrad=True)
    replay_memory = memory_replay.PrioritizedMemoryReplay(max_len=memory_replay_length)

    return env, optm, replay_memory, policy_network, target_network


def update_params(loss_fn = nn.SmoothL1Loss()):
    """This function updates the parameters of the policy network"""

    #If the agent has not accumulated enough experience yet
    if replay_memory.length() < config.BATCHSIZE:
        return
    
    #Sampling from the replay memory, samples is a batch of named tuples
    #Weights are from prioritized replay importance sampling
    samples, indices, weights = replay_memory.sample(config.BATCHSIZE)

    #Gathering all the states, actions, and rewards from sample
    batch = memory_replay.EnvStep(*zip(*samples))

    #Creating new tensors of concatenated values along the dimensions of the named tuple
    state_batch = torch.cat(batch.state).reshape(config.BATCHSIZE, 4, 84, 84).to(torch.float32)
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
    non_final_states = torch.cat(nf_states).reshape(len(nf_states), 4, 84, 84).to(torch.float32)
    with torch.no_grad():
        #Getting the predicted Q values from the target network, taking the max Q-value
        next_step_value_function[non_final_mask] = target_network(non_final_states).max(1).values

    #Now, compute the state-action Q function (with discount factor)
    next_q_value_function = (next_step_value_function * config.GAMMA) + reward_batch

    #The loss is the initial q value estimate subtracted by the estimate after one time step (after we gain more information)
    #q_value_function is the Q values of the actions we did take, next_q_val is the immediate reward + the max Q val of the next state
    loss = loss_fn(q_value_function, next_q_value_function.unsqueeze(1))
    loss_ret = (loss.detach()*weights) + 1e-5

    #clipping the loss to between (0,1) for stability
    loss = torch.clip(loss, min=-1, max=1)

    optm.zero_grad()
    loss.backward()

    #Updating transition priorities
    replay_memory.update_priorities(indices, loss_ret)

    #We clip the gradient in order to protect against exploding gradients
    torch.nn.utils.clip_grad_value_(policy_network.parameters(), 100)
    optm.step()


#Main training loop
if __name__ == '__main__':
    env, optm, replay_memory, policy_network, target_network = initialize_game("Pong-v0", optim.AdamW, memory_replay_length=10_000)

    """Pong Actions:
        0: Noop
        1: Fire
        2: Right
        3: Left
        4: Right Fire
        5: Left Fire

    """

    epsilon = config.EPS_START

    #State is an RGB image
    state, _ = env.reset()
    state = preprocess_frames(state)
    init = False

    #Stack the previous 4 frames as a new state
    frames_queue = deque([], maxlen=4)

    n_episode = 0
    episode_reward = 0

    #Running avwrage of reward for metric evaluation
    last_100_avg = [-21]
    score = deque(maxlen=100)

    for i in range(config.NUMFRAMES):

            #Adding dummy frames to stack of episodes at beginning of each episode
            if init == False:
                for i in range(4):
                    stack_frames(state, frames=frames_queue)
                init=True

            frames_tuple = tuple(tensor for tensor in frames_queue)
            stacked_frames = torch.stack(frames_tuple).reshape((4,84,84))
             
            with torch.no_grad():
                action = policy_network.act(state, epsilon, env)

            observation, reward, terminated, truncated, info = env.step(action.item())

            #Appending new preprocessed observation frame to queue
            frames_queue.append(preprocess_frames(observation))

            #Retrieving new state from updated frame_queue
            frames_tuple = tuple(tensor for tensor in frames_queue)
            next_state = torch.stack(frames_tuple).reshape((4,84,84))

            reward = torch.tensor([reward])
            done = terminated or truncated
            episode_reward += reward
            
            #If episode is about to terminate, do not append a next state
            if terminated:
                next_state = None

            replay_memory.add_episode(state, action, next_state, reward)

            #Replace the current state with the next observation from the environment
            state = next_state

            #update the policy network parameters
            update_params()

            #update the parameters of the target network every 10,000 frames
            if i % 10_000 == 0:
                target_sd = target_network.state_dict()
                policy_sd = policy_network.state_dict()

                for key in policy_sd:
                    target_sd[key] = (policy_sd[key] * config.SOFTUPDATE) + (target_sd[key] * (1-config.SOFTUPDATE))

                target_network.load_state_dict(target_sd)

            #Saving model every 100,000 frames
            if i % 100_000 == 0:
                weights_path = f'DQN_Atari/saved_models/model{i}.pth'

                # Save the model's state_dict (weights and biases)
                torch.save(target_network.state_dict(), weights_path)
                
            #If episode is over, break out of loop and iterate episode
            if done:
                #Decaying epsilon for lessened exploration
                epsilon = epsilon - (epsilon * config.EPS_DECAY)

                state, info = env.reset()
                state = preprocess_frames(state)

                init=False
                n_episode += 1
                score.append(episode_reward.item())
                next_state = None

                print(f'Episode {n_episode} is complete')
                print(f'Reward at this episode: {episode_reward}')
                episode_reward = 0

                if n_episode % 10_000 == 0:
                    last_100_avg.append(sum(score)/len(score))
                    plt.plot(np.arange(0,n_episode+1,10),last_100_avg)
                    plt.show()

