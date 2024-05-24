import gym
import replay_memory, neural_net, loss
import torch.optim as optim
import imageio
from collections import deque
import torch


if __name__ == '__main__':
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    obs_space = env.observation_space.shape[0]
    act_space = env.action_space.n
    #Programmable variables
    n_atoms = 51
    num_episodes = 5000
    steps = 0
    Vmin = -10
    Vmax = 10
    discount = 0.99
    lr = 0.0001
    batch_size = 32
    frames = deque(maxlen=10_000)

    memory = replay_memory.MemoryReplay(max_len=10_000)

    curr_net = neural_net.CategoricalDQN(obs_space, act_space, n_atoms, Vmin, Vmax)
    target_net = neural_net.CategoricalDQN(obs_space, act_space, n_atoms, Vmin, Vmax)
    optimizer = optim.Adam(curr_net.parameters(), lr=lr)

    for ep in range(num_episodes):
        state, _ = env.reset()
        rewards = 0

        while True:
            steps += 1
            action = curr_net.act(state)
            frame = env.render()
            frames.append(frame)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards += reward

            memory.remember(state, action, reward, obs, done)

            if memory.length() > batch_size:
                losses = loss.loss(Vmin, Vmax, n_atoms, discount, memory, batch_size, target_net, curr_net)
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            state = obs

            #Update target_net after 500 steps
            if steps % 500 == 0:
                sd = curr_net.state_dict()
                target_net.load_state_dict(sd)
                steps=0

            if done:
                if ep % 5 == 0:
                    print(f'Reward at episode {ep} is {rewards}')
                break

    imageio.mimwrite('lunar_lander.gif', frames[-20:], fps=60)




