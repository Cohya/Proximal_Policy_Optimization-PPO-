import matplotlib.pyplot as plt
import gym 
from PPO import  PPO
from Agent import Agent
from Actor_Critic_Nets import Actor, Critic
import tensorflow_probability as tfp
import numpy as np
import tensorflow as tf 
from utils import  watch_agent, record_agent


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

rewards_pretrain = agent.test(number_of_games = 500, env = env) # <--- parallel

agent.train(environment = 'CartPole-v1', NUM_OF_ENVS = 5, lr = 0.00025,aneal_lr= True,
            total_timesteps=50000, 
            use_threding=True)

rewards_postTrain= agent.test(number_of_games = 500, env = env) # <---  parallel

agent.save_weights()
# agent.load_weights(weight_file='Weights/weights.pickle')


plt.figure(10)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = '12'
plt.plot(rewards_postTrain, label = 'Post Trianing')
plt.plot(rewards_pretrain, label = 'Pre Train')
plt.xlabel('Episode', fontsize=16)
plt.ylabel('Cumulative rewards', fontsize=16)
plt.ylim([0,600])
plt.legend(frameon=False,loc = 2)


watch_agent(env, model = agent)
record_agent(env, model = agent)