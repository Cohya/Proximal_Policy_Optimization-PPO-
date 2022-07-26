
import tensorflow as tf 
import tensorflow_probability as tfp
import matplotlib.pyplot as plt 


class Agent():
    def __init__(self, actor_net, critic_net, model,  action_distribution = tfp.distributions.Categorical):
        
        self.model = model(actor_net = actor_net,
                           critic_net = critic_net,
                           distribution = action_distribution)
        
        self.critic = critic_net
        self.actor = actor_net
        self.distribution = action_distribution
        
        
    def get_value(self,x):
        return self.critic.call(x)
    
    def get_action(self, x):
        # x = np.asarray(x, dtype = np.float32)
        x = tf.expand_dims(tf.cast(x, dtype = tf.float32), axis = 0)
        logits = self.actor.call(x)
        probs = self.distribution(logits = logits)
        action = probs.sample()
        return action.numpy()[0] 
        
    def get_action_and_value(self, x, action = None):
        logits = self.actor.call(x)
        probs = self.distribution(logits = logits)
        
        if action is None:
            action  = probs.sample()
            
        critic = self.critic(x)

        return action, probs.log_prob(action), probs.entropy(), critic
    
    def test(self, number_of_games, env):
        
        rewards = []
        
        for game in range(number_of_games):
            s = env.reset()
            cummulative_reward = 0
            done = False
            while not done:
                action = self.get_action(s)
                next_obs, r, done, _ = env.step(action)
                cummulative_reward += r
                s = next_obs
            
            rewards.append(cummulative_reward)
            
        return rewards
        
    
    def watch_agent(self):
        pass
    
    def save_weights(self):
        pass
            
            
    def train(self, environment, NUM_OF_ENVS,num_of_steps = 128, lr = 0.00025,
              aneal_lr = True, total_timesteps = 25000 , num_of_mini_batches = 4,
              update_epochs = 4, gae =True, norm_advantage = True, clip_vloss = True,
              gamma = 0.99 ,lmbda = 0.95, entr_coef = 0.01,valu_coef = 0.25,
              use_gradient_norm_clipping = True, max_grad_norm = 0.5,clip_coef = 0.2,
              target_based_kl = None, verbose = True, plot_graphs = True, use_threding = True):
        
        
        (loss_vec, kl_vec, all_episodes_rewards, explained_var_vec) = self.model.train(
                                                                    environment = environment,
                                                                    NUM_OF_ENVS = NUM_OF_ENVS,
                                                                    num_of_steps = num_of_steps,
                                                                    lr = lr,
                                                                    aneal_lr = aneal_lr,
                                                                    total_timesteps = total_timesteps,
                                                                    num_of_mini_batches = num_of_mini_batches,
                                                                    update_epochs = update_epochs,
                                                                    gae = gae,
                                                                    norm_advantage = norm_advantage,
                                                                    clip_vloss = clip_vloss,
                                                                    gamma = gamma,
                                                                    lmbda = lmbda,
                                                                    entr_coef = entr_coef,
                                                                    valu_coef = valu_coef,
                                                                    use_gradient_norm_clipping = use_gradient_norm_clipping,
                                                                    max_grad_norm = max_grad_norm,
                                                                    clip_coef = clip_coef,
                                                                    target_based_kl = target_based_kl,
                                                                    verbose = verbose,
                                                                    use_threding = use_threding )
        
        if plot_graphs:
            plt.figure(1)
            plt.plot(loss_vec)
            plt.ylabel('loss')
            plt.xlabel('step')
            
            plt.figure(2)
            plt.plot(kl_vec)
            plt.ylabel(r'$KL-divergence between \frac{\pi_{\theta_{new}}}{\pi_{\theta_{old}}}$')
            plt.xlabel('step')
            
            plt.figure(3)
            plt.plot(all_episodes_rewards)
            plt.ylabel('Episodes Reward')
            plt.xlabel('step')
            
            plt.figure(4)
            plt.plot(explained_var_vec)
            plt.ylabel('explained_var_vec')
            plt.xlabel('step')
            
            
        
