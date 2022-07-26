
import gym 
from PPO import  PPO
from Agent import Agent
from Actor_Critic_Nets import Actor, Critic
import tensorflow_probability as tfp
import numpy as np
import tensorflow as tf 

env = gym.make('CartPole-v1')

observation_dims = env.observation_space.shape[0]
action_dims = env.action_space.n

actor = Actor(input_dims = (1,observation_dims), output_dims = action_dims)
critic = Critic(input_dims = (1,observation_dims), output_dims = 1)

model = PPO  # optimization technique

agent = Agent(actor_net = actor,
              critic_net = critic,
              model = model,
              action_distribution = tfp.distributions.Categorical)


agent.train(environment = 'CartPole-v1', NUM_OF_ENVS = 5, lr = 0.0002,aneal_lr= False,
            total_timesteps=30000, 
            use_threding=True)

rewards = agent.test(number_of_games = 100, env = env)

