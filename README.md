# Reinforcement Learning for Atari Games

## Purpose
This project aims to implement and compare two Reinforcement Learning (RL) algorithms, Deep Q Networks (DQN) and Advantage Actor-Critic (A2C), using PyTorch. The goal is to train agents to beat Atari games such as Pong, demonstrating the effectiveness of these algorithms in learning complex tasks.

## Table of Contents
- [Purpose](#purpose)
- [Skills Used](#skills-used)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Background](#background)

## Skills Used
- Reinforcement Learning Algorthms (DQN, A2C)
- Deep Learning (Pytorch Framework)
- Python Programming
- OpenAI Gym Environment

## Technologies Used
- Python
- PyTorch
- OpenAI Gym (for Atari game environments)

## Project
The project achieved the following results:
- Trained Deep Q Networks (DQN) agent to achieve high scores in Atari games, including Pong.
- Implemented prioritized experience replay in DQN for efficient sampling.
- Developed an Advantage Actor-Critic (A2C) agent that combines value-based and policy-based learning methods.
- Evaluated and compared the performance of DQN and A2C on Atari games.

## Background
### Deep Q Networks (DQN)
DQN is a value-based RL algorithm that learns to approximate the Q-value function. It uses a neural network to estimate the Q-values for each action in a given state. In this project, DQN is enhanced with a prioritized memory buffer, which improves learning efficiency by prioritizing experience replay based on temporal difference error.

### Advantage Actor-Critic (A2C)
A2C is a policy-based RL algorithm that combines actor and critic networks. The actor network learns a policy to select actions, while the critic network evaluates the state-value function. By leveraging both value-based and policy-based learning, A2C aims to improve sample efficiency and stability in training.

