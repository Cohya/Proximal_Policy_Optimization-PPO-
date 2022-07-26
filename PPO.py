
import tensorflow as tf 
from utils import EnvWrapper, EnvWrapperThread
import tensorflow_probability as tfp 
import numpy as np 
import time 
import matplotlib.pyplot as plt 
import gym 

def learning_rate_anealing(num_updates):
    def lr_anealing(number_of_update, lr_now):
        frac = 1.0 - (number_of_update - 1.0)/num_updates
        lr =  frac * lr_now
        return lr
    
    return lr_anealing

class PPO():
    def __init__(self, actor_net, critic_net, distribution = tfp.distributions.Categorical):
        self.critic_nnet = critic_net
        self.actor_nnet = actor_net
        self.distribution = distribution
        self.optimizer_base = tf.keras.optimizers.Adam
        
        # Collect all trainable params (weights)
        self.trainable_param = []
        self.trainable_param += self.critic_nnet.trainable_param
        self.trainable_param += self.actor_nnet.trainable_param
        
    def get_value(self,x):
        return self.critic_nnet.call(x)
    
    def get_action_and_value(self, x, action = None):
        logits = self.actor_nnet.call(x)
        probs = self.distribution(logits = logits)
        
        if action is None:
            action  = probs.sample()
            
        critic = self.critic_nnet(x)

        return action, probs.log_prob(action), probs.entropy(), critic
    
    
    def train(self, environment, NUM_OF_ENVS,num_of_steps = 128, lr = 0.00025,
              aneal_lr = True, total_timesteps = 25000 , num_of_mini_batches = 4,
              update_epochs = 4, gae =True, norm_advantage = True, clip_vloss = True,
              gamma = 0.99 ,lmbda = 0.95, entr_coef = 0.01,valu_coef = 0.25,
              use_gradient_norm_clipping = True, max_grad_norm = 0.5,clip_coef = 0.2,
              target_based_kl = None, verbose = True, use_threding = True):
        """
        environment - > gym.make('CartPole-v1') (object)
        NUM_OF_ENVS - numebr of parallel environment to use during training (int) 
        num_of_steps -> the number of steps before each training loop (int) 
        lr -> leaninig rate for the optimizer (float32)
        aneal_lr -> if you with to decrease the lr during training (Boolean) 
        total_timesteps - > number of steps that the agent performe over the environment(int)
        num_of_mini_batches - > number of minibatches during trainig at each training loop (int)
        update_epochs -> number of epoch during each training step 
        gae -> Generel advantage estimation (Boolean)
        norm_advantage -> if you wish to performe advantage normalization  ((Boolean))
        clip_vloss -> if you wish to clip the value loss size to avoid high gradients ((Boolean))
        gamma -> discount factor (float in [0,1])
        lmbda -> use for the gae
        entr_coef -> the costant which multiply entropy in the loss fucntion (c2 constant in the original PPO paper) (float)
        vf_coef ->  the costant which multiply MSE of the value_loss in the loss fucntion (c1 constant in the original PPO paper) (float)
        use_gradient_norm_clipping -> if you wish to performe normclipping to the gradients values (Boolean)
        max_grad_norm -> the maximum value of the norm cliping 
        clip_coef  --> the coeficient of the clip loss 
        target_based_kl --> to stop gradient update if the KL-divergence is too high (which means that the new distribution is to far), 
                            provide lockModes (float, you can use 0.015)
        verbose -> set to true if you whish to get information during training (Boolean)
        use_threding -> is you want to apply parallel computing based threading over the steps 
        """
        
        self.use_gradient_norm_clipping  = use_gradient_norm_clipping 
        self.max_grad_norm = max_grad_norm
        self.clip_vloss = clip_vloss
        self.entr_coef = entr_coef
        self.valu_coef = valu_coef
        self.optimizer = self.optimizer_base(learning_rate = lr, epsilon=1e-5)
        # Create environments
        envs_vec = [gym.make(environment) for i in range(NUM_OF_ENVS)]
        if use_threding:
            envs = EnvWrapperThread(envs_vec = envs_vec)
        else:
            envs = EnvWrapper(envs_vec = envs_vec)
        
        observation_dims = envs_vec[0].observation_space.shape[0] 
        # Creat a store for what we need 
        obs_vec = np.zeros(shape = (num_of_steps, NUM_OF_ENVS, observation_dims))
        actions_vec = np.zeros(shape = (num_of_steps, NUM_OF_ENVS), dtype=np.int32) - 1 
        logprobs_vec = np.zeros(shape = (num_of_steps, NUM_OF_ENVS)) 
        rewards_vec = np.zeros(shape = (num_of_steps, NUM_OF_ENVS))
        dones_vec = np.zeros(shape = (num_of_steps, NUM_OF_ENVS))
        values_vec = np.zeros(shape = (num_of_steps, NUM_OF_ENVS))
        
        global_step = 0
            
        start_time = time.time()
        
        # reset all games 
        obs = envs.reset() # <- it is a matrix (NUM_OF_ENVS x  env.observation_space.sahpe[0])
        done = [False for e in envs_vec] # initialized all dones to False to statrt the game 
        
        batch_size = NUM_OF_ENVS * num_of_steps
        num_updates = total_timesteps // batch_size
        mini_batch_size = batch_size//num_of_mini_batches
        n_batches = batch_size // mini_batch_size
        
        if aneal_lr:
            anealFunc = learning_rate_anealing(num_updates = num_updates)
            
        
        # Collect all usefull information 
        all_episodes_rewards = []
        explained_var_vec = []
        kl_vec = []
        loss_vec = []
        
        # start the trainibg 

        for update in range(1, num_updates+1):
            # Anealing lr step 
            
            if aneal_lr:
                lr_new = anealFunc(update,lr)
                self.optimizer.learning_rate.assign(lr_new)
                
            for step in range(num_of_steps):
                global_step += 1 * NUM_OF_ENVS
                obs_vec[step] = obs
                dones_vec[step] = np.asarray(done)
                
                action, logprob, _, value = self.get_action_and_value(obs_vec[step])
                action = tf.stop_gradient(action).numpy()
                actions_vec[step] =  action
                logprobs_vec[step] = tf.stop_gradient(logprob)
                values_vec[step] = tf.stop_gradient(tf.transpose(value))
                
                # Now performe a step in each env
                obs_next, r, done,info = envs.step(action)
                obs = obs_next
                
                rewards_vec[step] = r
                
                if verbose :
                    for item in info:
                        if "steps" in item.keys() and step % 1 ==0:
                            all_episodes_rewards.append(item['rewards'])
                            if step % 50 ==0:
                                print(f"global_step = {global_step }, episode steps={item['steps']} , episodic_return={item['rewards']}")
                            
                            
            ### No gradients now (collect data fro training from experience)
            next_value = tf.transpose(self.get_value(obs))
            if gae: # General advantage estimation  
                advantages = np.zeros_like(rewards_vec)
                lastgaelam = 0
                for t in reversed(range(num_of_steps)):
                    if t == num_of_steps - 1:
                        nextnonterminal = 1.0 - done # creating a mask 
                        nextvalues = next_value.numpy()
                        
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
            batch_obs = np.reshape(obs_vec, newshape=(num_of_steps * NUM_OF_ENVS,observation_dims)) # flatten 
            batch_logprobs = np.reshape(logprobs_vec, newshape=(num_of_steps*NUM_OF_ENVS))
            batch_actions = np.reshape(actions_vec, newshape = (num_of_steps * NUM_OF_ENVS))
            batch_advatages = np.reshape(advantages, newshape= (num_of_steps*NUM_OF_ENVS))
            batch_returns = np.reshape(returns_vec, newshape = (num_of_steps * NUM_OF_ENVS))
            batch_values = np.reshape(values_vec, newshape = (num_of_steps * NUM_OF_ENVS))
            
            batch_indexs = np.arange(batch_size) # 
            self.clipfracs = []
            
            for epoch in range(update_epochs):
                np.random.shuffle(batch_indexs)
                for j in range(n_batches):
                    mini_batch_idx = batch_indexs[j * mini_batch_size : (j+1) * mini_batch_size]
                    
                    mini_batch_observation = batch_obs[mini_batch_idx]
                    mini_batch_actions = batch_actions[mini_batch_idx]
                    mini_batch_logprobs = batch_logprobs[mini_batch_idx]
                    mini_batch_advatages = batch_advatages[mini_batch_idx]
                    mini_batch_returns = batch_returns[mini_batch_idx]
                    mini_batch_values = batch_values[mini_batch_idx]
                    
                    if norm_advantage:
                        mini_batch_advatages =  (mini_batch_advatages - np.mean(mini_batch_advatages)) / (np.std(mini_batch_advatages) + 1e-8)
                        
                    loss , approx_kl = self.update_weights(mini_batch_observation = mini_batch_observation,
                                                           mini_batch_actions = mini_batch_actions,
                                                           mini_batch_logprobs = mini_batch_logprobs,
                                                           mini_batch_advatages = mini_batch_advatages,
                                                           mini_batch_returns = mini_batch_returns,
                                                           mini_batch_values = mini_batch_values,
                                                           clip_coef  = clip_coef)
                    
                    kl_vec.append(approx_kl)
                    loss_vec.append(loss)
                if target_based_kl is not None:
                  if approx_kl > target_based_kl:
                     break
                 
            y_pred, y_true = batch_values, batch_returns
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            if verbose:
                print("Steps/secodes:", int(global_step / (time.time() - start_time)))
            explained_var_vec.append(explained_var)  
            
        return loss_vec, kl_vec, all_episodes_rewards, explained_var_vec
                
    def update_weights(self, mini_batch_observation,mini_batch_actions,mini_batch_logprobs,
                       mini_batch_advatages,mini_batch_returns, mini_batch_values, clip_coef):
        
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            _, newlogprob, entropy, newvalues = self.get_action_and_value(x = mini_batch_observation ,
                                                                           action=mini_batch_actions)
            
            logratio = newlogprob - mini_batch_logprobs 
            ratio = tf.math.exp(logratio) # first epoch should be all one's (it is the same weights)
            
            ### just for understanding what happens (watch KL distance between Policy(theta_old) and Policy(theta_new)) 
            #old_approx_kl = tf.reduce_mean(-logratio)
            approx_kl = tf.reduce_mean((ratio-1.0) - logratio) # Estimator of KL-divergence between KL(Policy(theta_new) = x|Policy(theta_old)=y)  = sum (x log(x/y)) 
            self.clipfracs += [np.mean(np.int32(tf.math.abs(ratio - 1.0) > clip_coef))]#to ccount how mant clliping we performed 
            
            # Policy loss (we want to miximize it so we take it in minus in the global cost)
            pg_loss1 = mini_batch_advatages * ratio
            pg_loss2 = mini_batch_advatages * tf.clip_by_value(ratio, 1-clip_coef, 1+clip_coef)
            pg_loss = tf.reduce_mean(tf.minimum(pg_loss1, pg_loss2))
            
            # Value loss (Critic)
            newvalues = tf.squeeze(newvalues, axis = 1)
            
            if self.clip_vloss:
                value_loss_unclipped = tf.math.square(newvalues - mini_batch_returns )
                value_clipped = ( mini_batch_values + 
                                    tf.clip_by_value(newvalues - mini_batch_values,
                                                     -clip_coef,
                                                     clip_coef)
                                    )
                value_loss_clipped = (value_clipped - mini_batch_returns)**2
                
                value_loss_max = tf.maximum(value_loss_unclipped, value_loss_clipped)# check if it shoud be min 
                value_loss = tf.reduce_mean(value_loss_max)
            else:
                value_loss = tf.reduce_mean((newvalues -  mini_batch_returns)**2)
            
            entropy_loss = tf.reduce_mean(entropy) # we watn to miximize it to provide exploration 
            loss = - pg_loss - self.entr_coef * entropy_loss + value_loss *  self.valu_coef
        
        gradients = tape.gradient(loss, self.trainable_param)
        # print(gradients)
        # clip the gradient using norm (to avoid high gradients / exploding gradients)
        if self.use_gradient_norm_clipping :
            gradients_clipped =[tf.clip_by_norm(g, self.max_grad_norm)for g in gradients]
            
        self.optimizer.apply_gradients(zip(gradients_clipped, self.trainable_param))
        
        return loss , approx_kl
            
        

    
