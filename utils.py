
import numpy as np
from threading import Thread, Lock
import time 
from gym import wrappers
import os 


class EnvWrapper():
    def __init__(self, envs_vec):
        
        self.envs = envs_vec
        self.dones = [False for _ in self.envs]
        self.num_envs = len(envs_vec)
        self.cummulative_reward = [0 for _ in envs_vec]
        self.cummulative_steps = [0 for _ in envs_vec]
    def step(self, action):
        all_ifo = []
        # default = [None, None, None, None], 0, True, {}
        # all_ifo = np.asarray([env.step(a) if not done else default for a,env,done in 
        #                       zip(action, self.envs, self.dones)])
        for i in range(self.num_envs):
            # if not self.dones[i]:
            env = self.envs[i]
            a = action[i]
            obs, r, done, info = env.step(a)
            self.cummulative_reward[i] += r
            self.cummulative_steps[i] += 1

            if done :
                info['steps'] = self.cummulative_steps[i]
                info['rewards'] = self.cummulative_reward[i]
                obs = env.reset()
                self.cummulative_steps[i] = 0
                self.cummulative_reward[i] = 0
                
            all_ifo.append([ obs, r, done, info ])
                
        all_ifo = np.asarray(all_ifo)        
        obs = np.vstack(all_ifo[:,0])
        r = np.reshape(all_ifo[:,1],newshape=(self.num_envs))
        dones = np.reshape(all_ifo[:,2], newshape=(self.num_envs))
        info = np.reshape(all_ifo[:,3], newshape=(self.num_envs))
        
        # self.dones = dones
        
        return obs, r, dones, info
    
    def reset(self):
        obs = np.asarray([e.reset() for e in self.envs])
        return obs
    

class EnvWrapperThread():
    def __init__(self, envs_vec):
        self.envs = envs_vec
        self.dones = [False for _ in self.envs]
        self.num_envs = len(envs_vec)
        self.cummulative_reward = [0 for _ in envs_vec]
        self.cummulative_steps = [0 for _ in envs_vec]
        self.mutex = Lock()
    def step_warpper(self,env, a, all_ifo, i):
        
        obs, r, done, info = env.step(a)
        # you must lock before you add to general memo
        # self.mutex.acquire()
        self.cummulative_reward[i] += r
        self.cummulative_steps[i] += 1
        
        if done :
            info['steps'] = self.cummulative_steps[i]
            info['rewards'] = self.cummulative_reward[i]
            obs = env.reset()
            self.cummulative_steps[i] = 0
            self.cummulative_reward[i] = 0
            
        self.all_ifo.append([ obs, r, done, info ])
        # self.mutex.release()
            
    def step(self, action):
        self.all_ifo = []
        # default = [None, None, None, None], 0, True, {}
        # all_ifo = np.asarray([env.step(a) if not done else default for a,env,done in 
        #                       zip(action, self.envs, self.dones)])
        for i in range(self.num_envs):
            # if not self.dones[i]:
            env = self.envs[i]
            a = action[i]
            t = Thread(target = self.step_warpper, 
                       args = ([env,a, self.all_ifo, i]))
            t.start()
            t.join()
                
        all_ifo = np.asarray(self.all_ifo)  
        # print(all_ifo)
        obs = np.vstack(all_ifo[:,0])
        r = np.reshape(all_ifo[:,1],newshape=(self.num_envs))
        dones = np.reshape(all_ifo[:,2], newshape=(self.num_envs))
        info = np.reshape(all_ifo[:,3], newshape=(self.num_envs))
        
        return obs, r, dones, info
    
    def reset(self):
        obs = np.asarray([e.reset() for e in self.envs])
        return obs


## for anealing earning rate 
def learning_rate_anealing(num_updates):
    def lr_anealing(number_of_update, lr_now):
        frac = 1.0 - (number_of_update - 1.) /num_updates
        lr =  frac * lr_now
        return lr
    
    return lr_anealing


def watch_agent(env, model, image_transformer = None):
    obs = env.reset()
    
    # state = np.stack([obs_small] * 4 , axis = 2) #we creat the first state s0
    done = False
    episode_reward = 0
    if image_transformer is None :
       state = obs
    else:
       state = image_transformer.transform(obs)
       
    while not done:
        env.render()
        action = model.get_action(state)
        obs, reward, done, info = env.step(action)
        if image_transformer is None :
            state = obs
        else:
            state = image_transformer.transform(obs)
        
        # Compute total reward
        episode_reward += reward
        state = obs
        time.sleep(0.02)
    
    env.close()
    
def record_agent(env, model,image_transformer = None, videoNum= 0):
    if not os.path.isdir('./videos'):  
        os.makedirs('videos')
   
    dire = './videos/' + 'vid_' + str(videoNum)
    env = wrappers.Monitor(env, dire)
    obs = env.reset()
    if image_transformer is None:
        state = obs 
    else:    
        state = image_transformer.transform(obs)
    # state = np.stack([obs_small] * 4, axis = 2)
    done = False
    episode_reward = 0
    
    while not done:
        action = model.get_action(state)
        obs, reward, done, info = env.step(action)
        if image_transformer is None:
            state  = obs
        else:    
            state = image_transformer.transform(obs)
        
        # next_state = update_state(state, obs_small)
        
        episode_reward += reward
        # state = next_state
        time.sleep(0.02)
                
    print("record video game in folder video %s / " % 'vid_' + str(videoNum), "episode reward: ", episode_reward)
# import gym 
# envs = [gym.make('CartPole-v1') for i in range(4)]
# envswr = EnvWrapperThread(envs)
# envswr.reset()
# obs, r, dones, info = envswr.step([0,0,0,0])