
import numpy as np
from threading import Thread, Lock

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



# import gym 
# envs = [gym.make('CartPole-v1') for i in range(4)]
# envswr = EnvWrapperThread(envs)
# envswr.reset()
# obs, r, dones, info = envswr.step([0,0,0,0])