import gym
import config as fig, agent as a
from wrappers import make_env

if __name__ == '__main__':
    env = make_env("PongNoFrameskip-v4")
    seed = 42  # You can use any integer you like
    env.action_space.seed(42)

    obs_space = env.observation_space.shape[0]
    act_space = env.action_space.n

    agent = a.Agent((4,84,84), act_space, fig.BATCHSIZE)
    state, _ = env.reset(seed=42)

    for episode in range(fig.NUMEPISODES):
        state, _ = env.reset()
        steps = 0
        rewards = 0
        done = False

        while not done:
            #Run policy πθold in environment for T timesteps

            #log_prob stored for πθold calculation, state_value stored for baseline subtraction in advantage
            action, log_prob, state_value = agent.act(state)

            steps += 1

            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rewards += reward

            agent.collect(reward, action, state_value, log_prob, state, done)

            state = observation

            if fig.TRAJLENGTH % steps == 0:

                #At end of each trajectory, run optimization for k epochs
                for epoch in range(fig.OPTM_EPOCHS):
                    agent.learn()

        #Clear the agent's memory after each trajectory
        agent.clear()

        print(f'Reward at episode {episode} is {rewards}')
