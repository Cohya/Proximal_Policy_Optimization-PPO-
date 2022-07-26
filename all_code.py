import tensorflow as tf 
import numpy as np
import tensorflow_probability as tfp
import gym 
import time 
import torch  
import matplotlib.pyplot as plt 
from utils import EnvWrapper
from Actor_Critic_Nets import Actor, Critic
NUM_OF_ENVS = 4
num_of_steps = 128
clip_coef = 0.2

def learning_rate_anealing(num_updates):
    def lr_anealing(number_of_update, lr_now):
        frac = 1.0 - (number_of_update - 1.) /num_updates
        lr =  frac * lr_now
        return lr
    
    return lr_anealing

class Agent():
    def __init__(self, criticNet, actorNet, distribution = tfp.distributions.Categorical):
        
        self.critic = criticNet
        self.actor = actorNet
        self.distribution = distribution
        
        self.trainable_param = []
        self.trainable_param += self.critic.trainable_param
        self.trainable_param += self.actor.trainable_param
        
        
    def get_value(self,x):
        return self.critic.call(x)
    
    def get_action_and_value(self, x, action = None):
        logits = self.actor.call(x)
        probs = self.distribution(logits = logits)
        
        if action is None:
            action  = probs.sample()
            
        critic = self.critic(x)

        return action, probs.log_prob(action), probs.entropy(), critic
 

    
    def reset(self):
        obs = np.asarray([e.reset() for e in self.envs])
        return obs
    
    
# Creat several environments
envs_vec = [gym.make('CartPole-v1') for i in range(NUM_OF_ENVS)]
envs = EnvWrapper(envs_vec = envs_vec)
# Prepare the agent
observations_dims = envs_vec[0].observation_space.shape[0] 
input_dims = (1, envs_vec[0].observation_space.shape[0])
action_space = envs_vec[0].action_space.n
agent = Agent(criticNet=Critic(input_dims=input_dims, output_dims=1), 
              actorNet=Actor(input_dims= input_dims, output_dims= action_space),
              distribution=tfp.distributions.Categorical)

# Buid an optimizaer 
lr= 0.00025
optimizer = tf.keras.optimizers.Adam(learning_rate =lr, epsilon=1e-5)


        
## for anealing earning rate 


# Creat a store for what we need 
obs_vec = np.zeros(shape = (num_of_steps, NUM_OF_ENVS, envs_vec[0].observation_space.shape[0]))
actions_vec = np.zeros(shape = (num_of_steps, NUM_OF_ENVS), dtype=np.int32) - 1 
logprobs_vec = np.zeros(shape = (num_of_steps, NUM_OF_ENVS)) 
rewards_vec = np.zeros(shape = (num_of_steps, NUM_OF_ENVS))
dones_vec = np.zeros(shape = (num_of_steps, NUM_OF_ENVS))
values_vec = np.zeros(shape = (num_of_steps, NUM_OF_ENVS))

### Start the game 
global_step = 0 # to decrease the learning rate 
start_time = time.time()
obs = envs.reset() # reset all games
done = [False for e in envs_vec] # initialized all dones to False to statrt the game 

total_timesteps = 25000
num_of_mini_batches = 4
batch_size = NUM_OF_ENVS * num_of_steps
num_updates = total_timesteps // batch_size
mini_batch_size = batch_size//num_of_mini_batches
n_batches = batch_size // mini_batch_size

aneal_lr = True #False
gae =True
norm_advantage = True
clip_vloss = True
target_kl = None # 0.015

gamma = 0.99 # discount factor 
lmbda = 0.95
ent_coef = 0.01
max_grad_norm = 0.5
vf_coef = 0.25# 0.5


if aneal_lr:
    anealFunc = learning_rate_anealing(num_updates = num_updates)
 
all_episodes_rewards = []
for update in range(1, num_updates + 1):
    # Aneaing the learning rate 
    if aneal_lr: 
        lr_now = anealFunc(update, lr)
        optimizer.learning_rate.assign(lr_now)   

    # Start playing 
    for step in range(num_of_steps):
        global_step += 1 * NUM_OF_ENVS
        obs_vec[step] = obs
        dones_vec[step] = np.asarray(done)
        
        
        action, logprob, _, value = agent.get_action_and_value(obs_vec[step])
        action = tf.stop_gradient(action).numpy()
        actions_vec[step] = action
        logprobs_vec[step] = tf.stop_gradient(logprob)
        values_vec[step] = tf.stop_gradient(tf.transpose(value))
        
        # Now performe a step in each env
        obs_next, r, done,info = envs.step(action)
        obs = obs_next
        
        rewards_vec[step] = r
   
        for item in info:
            if "steps" in item.keys() and step % 10 ==0:
                print(f"global_step = {global_step }, episode steps={item['steps']} , episodic_return={item['rewards']}")
                all_episodes_rewards.append(item['rewards'])
                # break
                
    ### No gradients now
    next_value = tf.transpose(agent.get_value(obs)).numpy()
    if gae: # General advantage estimation  
        advantages = np.zeros_like(rewards_vec)
        lastgaelam = 0
        for t in reversed(range(num_of_steps)):
            if t == num_of_steps - 1:
                nextnonterminal = 1.0 - done # creating a mask 
                nextvalues = next_value
                
            else:
                nextnonterminal = 1.0 - dones_vec[t + 1]
                nextvalues = values_vec[t + 1]
                
            delta = rewards_vec[t] + gamma * nextvalues * nextnonterminal - values_vec[t]
            advantages[t] = delta + gamma * lmbda * nextnonterminal * lastgaelam 
            lastgaelam  = advantages[t]
        returns_vec = advantages + values_vec
        
    else:
        returns_vec = np.zeros_like(values_vec)
        
        for t in reversed(range(num_of_steps)):
            if t == num_of_steps -1 :
                # means this is the first 
                nextnonterminal = 1.0 - done
                next_return = next_value # we use semi gradinent and use thr last estimation
            else:
                nextnonterminal = 1.0 - dones_vec[t+1]
                next_return = returns_vec[t+1]
            
            returns_vec[t] = rewards_vec[t] + gamma * nextnonterminal * next_return
        advantages =  returns_vec - values_vec
        
    # Flatten the batch 
    batch_obs = np.reshape(obs_vec, newshape=(num_of_steps * NUM_OF_ENVS,observations_dims)) # flatten 
    batch_logprobs = np.reshape(logprobs_vec, newshape=(num_of_steps*NUM_OF_ENVS))
    batch_actions = np.reshape(actions_vec, newshape = (num_of_steps * NUM_OF_ENVS))
    batch_advatages = np.reshape(advantages, newshape= (num_of_steps*NUM_OF_ENVS))
    batch_returns = np.reshape(returns_vec, newshape = (num_of_steps * NUM_OF_ENVS))
    batch_values = np.reshape(values_vec, newshape = (num_of_steps * NUM_OF_ENVS))
 
    b_inds = np.arange(batch_size) # 
    clipfracs = []
    update_epochs = 4
    for epoch in range(update_epochs):
        np.random.shuffle(b_inds)
        for j in range(n_batches):
            mini_batch_idx = b_inds[j*mini_batch_size:(j+1)*mini_batch_size]
            
            mini_batch_observation = batch_obs[mini_batch_idx]
            mini_batch_actions = batch_actions[mini_batch_idx]
            mini_batch_logprobs = batch_logprobs[mini_batch_idx]
            mini_batch_advatages = batch_advatages[mini_batch_idx]
            mini_batch_returns = batch_returns[mini_batch_idx]
            mini_batch_values = batch_values[mini_batch_idx]
            
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                _, newlogprob, entropy, newvalues = agent.get_action_and_value(x =mini_batch_observation ,
                                                                               action=mini_batch_actions)
                logratio = newlogprob - mini_batch_logprobs# here batch_logprobs[mb_inds]
                
                ratio = tf.math.exp(logratio) # forst epoch should be all one's (it is the same weights)
                
                ### just for understanding what happens 
                old_approx_kl = tf.reduce_mean(-logratio)
                approx_kl = tf.reduce_mean((ratio-1.0) - logratio)
                clipfracs += [np.mean(np.int32(tf.math.abs(ratio - 1.0) > clip_coef))]#to ccount how mant clliping we performed 
                
                # mini_batch_advatages = batch_advatages[mb_inds] # here
                
                if norm_advantage:
                    mini_batch_advatages =  (mini_batch_advatages - np.mean(mini_batch_advatages)) / (np.std(mini_batch_advatages) + 1e-8)
                    
                # Policy loss (we want to miximize it so we take it in minus in the global cost)
                pg_loss1 = mini_batch_advatages * ratio
                pg_loss2 = mini_batch_advatages * tf.clip_by_value(ratio, 1-clip_coef, 1+clip_coef)
                pg_loss = tf.reduce_mean(tf.minimum(pg_loss1, pg_loss2))
                
                # Value loss (Critic)
                newvalues = tf.squeeze(newvalues, axis = 1)
                
                if clip_vloss:
                    value_loss_unclipped = tf.math.square(newvalues - mini_batch_returns )
                    value_clipped = ( mini_batch_values + 
                                        tf.clip_by_value(newvalues - mini_batch_values,
                                                         -clip_coef,
                                                         clip_coef)
                                        )
                    value_loss_clipped = (value_clipped - mini_batch_returns)**2
                    
                    value_loss_max = tf.maximum(value_loss_unclipped, value_loss_clipped)
                    value_loss = tf.reduce_mean(value_loss_max)
                else:
                    value_loss = tf.reduce_mean((newvalues -  mini_batch_returns)**2)
                
                entropy_loss = tf.reduce_mean(entropy) # we watn to miximize it to provide exploration 
                loss = - pg_loss - ent_coef * entropy_loss + value_loss * vf_coef #* 0.5

        
            grads= tape.gradient(loss, agent.trainable_param)
            # clip the gradient using norm (to avoid high gradients)
            gradients_clipped =[tf.clip_by_norm(g, max_grad_norm)for g in grads]
            optimizer.apply_gradients(zip(gradients_clipped, agent.trainable_param))
    
        if target_kl is not None:
            if approx_kl > target_kl:
                break
  
    y_pred, y_true = batch_values, batch_returns
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    print("SPS:", int(global_step / (time.time() - start_time)))
            #     break
        # break
    # break
        
    
        
plt.plot(all_episodes_rewards)

 


# def f(x):
#     return x**2


# x = tf.Variable(99.0)
# for i in range(1):
#     with tf.GradientTape() as tape:
#        with tf.stop_gradient():
#            gg = f(x)
           
#        y = f(x) + gg
    
#     gradients = tape.gradient(y, x)
    # optimizer.apply_gradients(zip([gradients], [x]))
    # lr = optimizer.learning_rate 
    # optimizer.learning_rate.assign(max(lr - 0.001, 0.1))

        
    
        