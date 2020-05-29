#!/usr/bin/env python
# coding: utf-8

# define gym environment for RL

# In[1]:


import torch
import multiprocessing
import numpy as np
import pickle
import time
from common import find_grid_idx, extract_time_feat, min_gps, max_gps, real_distance, block_number, cuda
delta_gps = max_gps - min_gps


# In[2]:


pkl_folder = 'data/pkl/'
rd_data = pickle.load(open(pkl_folder + 'grid_order_reward.pkl', 'rb'))
rd_data = np.concatenate([np.expand_dims(rd_data['reward'],2), np.expand_dims(rd_data['order'],2)], axis=2)


# In[3]:


action_data = pickle.load(open(pkl_folder + 'carenv_actions.pkl', 'rb'))


# In[31]:


class CarEnv:
    """KDDCup2020 car environment
    State: [lat, lon, hour, average_reward, demand]
    Action: [lat, lon, ETA, reward, prob]
    
    Args:
        actions: (block_number, 24) list, every item contains actions
                 [startlat, startlon, endlat, endlon, ETA, reward, prob] with shape (x,) except prob 
                 with shape (x,10)
        reward_demand: shape (block_number, 2), means average reward and average demand    TODO: use NN to predict?
        random_seed: initial seed
        choose_ratio: select how much actions
        choose_max: select as maximum how much actions
    """
    
    def __init__(self, actions, reward_demand, random_seed = 0, choose_ratio = 0.6, choose_max = 12):
        self.actions = actions
        self.reward_demand = reward_demand
        self.rng = np.random.RandomState(random_seed)
        self.now_time = None
        self.now_state = None
        self.now_actions = None
        self.choose_ratio = choose_ratio
        self.choose_max = choose_max
        self.fail_waste = 600.0
        self.is_reset = False
        #self.reset()
        
    def _get_reward_demand(self, s):
        return self.reward_demand[int(s[0] * block_number[0]) * block_number[1] + int(s[1] * block_number[1]), s[2]]
        
    def _select_action(self):
        default_action = [np.array([self.now_state[0]]), np.array([self.now_state[1]]), 
                          np.array([self.now_state[0]]), np.array([self.now_state[1]]), 
                          np.array([0]), np.array([0]), np.array([1.0])]
        pos = (block_number * self.now_state[:2]).astype(int)
        expand = 1
        t_expand = 1
        alla = []
        for i in range(-expand, expand + 1):
            for j in range(-expand, expand + 1):
                for k in range(-t_expand, t_expand + 1):
                    k = (k + self.now_state[2] + 24) % 24
                    nowp = [i, j] + pos
                    if (nowp < 0).any() or (nowp >= block_number).any():
                        continue
                    block_idx = (nowp * [block_number[1], 1]).sum()
                    #print(block_idx, k)
                    if len(self.actions[block_idx][k][0]) > 0:
                        alla.append(self.actions[block_idx][k])
        alla = [np.concatenate(x) for x in zip(*alla)]
        if len(alla[0]) == 0:
            return [default_action]
        choose_num = int(self.rng.normal(self.choose_ratio, 2) * len(alla[0]))
        if choose_num <= 0:
            choose_num = 1
        if choose_num > len(alla[0]):
            choose_num = len(alla[0])
        if choose_num >= self.choose_max:
            choose_num = self.choose_max - 1
        choose = self.rng.choice(len(alla[0]), choose_num, replace = False)
        alla = [x[choose] for x in alla]
        gps = np.stack(alla[:2]).transpose(1, 0)
        gps = (gps - self.now_state[:2]) * real_distance
        gps = (gps ** 2).sum(axis=1) ** 0.5
        alla[4] += (gps / 5.7).astype(int) # add time driving to there
        gps = (gps / 200).astype(int)
        gps[gps > 9] = 9
        alla[-1] = np.choose(gps, alla[-1].T)
        #print(gps, alla[-1])
        for i in range(len(alla)):
            alla[i] = np.append(alla[i], default_action[i])
        return alla
    
    def reset(self):
        while True:
            self.now_time = self.rng.randint(24) * 3600
            s = [*self.rng.random(2), time.localtime(self.now_time).tm_hour]
            s += self._get_reward_demand(s).tolist()
            self.now_state = s
            a = self._select_action()
            if len(a) == 1:
                continue
            self.now_actions = a
            break
        self.is_reset = True
        return self.now_state, {'time': self.now_time}
    
    def step(self, action):
        assert(self.is_reset)
        a = [x[action] for x in self.now_actions][2:]
        if self.rng.random() < a[4]:
            reward = 0
            length = self.fail_waste # waste some time
            self.now_time += length
            self.now_state = [*self.now_state[:2], time.localtime(self.now_time).tm_hour]
            s = self.now_state
            s += self._get_reward_demand(s).tolist()
            self.now_actions = self._select_action()
            return s, reward, length, {'time': self.now_time}
        reward = a[3]
        length = a[2]
        self.now_time = self.now_time + length
        self.now_state = [*a[:2], time.localtime(self.now_time).tm_hour]
        s = self.now_state
        s += self._get_reward_demand(s).tolist()
        self.now_actions = self._select_action()
        return s, reward, length, {'time': self.now_time}
    def get_actions(self):
        assert(self.is_reset)
        return self.now_actions


# In[32]:


class EnvWorker(multiprocessing.Process):
    def __init__(self, env, envargs, pipe1, pipe2):
        multiprocessing.Process.__init__(self, daemon = True)
        self.env = env(*envargs)
        self.pipe = pipe1
        self.pipe2 = pipe2
    
    def run(self):
        self.pipe2.close()
        while True:
            try:
                cmd, data = self.pipe.recv()
                if cmd == 'step':
                    self.pipe.send(self.env.step(data))
                elif cmd == 'get_actions':
                    self.pipe.send(self.env.get_actions())
                elif cmd == 'close':
                    self.pipe.close()
                    break
                elif cmd == 'reset':
                    self.pipe.send(self.env.reset())
                else:
                    raise NotImplementedError
            except EOFError:
                break
                
class EnvVecs:
    def __init__(self, env_class, n_envs, env_args, arg_seed_pos = -1, seed = 0):
        self.waiting = False
        self.closed = False

        self.remotes, self.work_remotes = zip(*[multiprocessing.Pipe(duplex=True) for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            #args = (env_class, work_remote, remote)
            # daemon=True: if the main process crashes, we should not cause things to hang
            args = list(env_args)
            if arg_seed_pos != -1:
                args[arg_seed_pos] = seed
                seed += 1
            process = EnvWorker(env_class, args, work_remote, remote)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()
        self.is_reset = False

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, lengths, infos = zip(*results)
        self._get_actions()
        return self._flatten_obs(obs), np.stack(rews), np.stack(lengths), self._flatten_info(infos)

    def step(self, actions):
        assert(self.is_reset)
        self.step_async(actions)
        return self.step_wait()
    
    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True
    
    def _get_actions(self):
        for remote in self.remotes:
            remote.send(('get_actions', None))
        self.actions = [remote.recv() for remote in self.remotes]

    def get_actions(self):
        assert(self.is_reset)
        return self.actions
        
    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        self.is_reset = True
        self._get_actions()
        return self._flatten_obs(obs), self._flatten_info(infos)
    
    def _flatten_obs(self, obs):
        #print(obs)
        obs = list(zip(*obs))
        obs = list(map(lambda x:np.stack(x), obs))
        #print(obs)
        return obs
    
    def _flatten_info(self, info):
        if len(info) == 0:
            return {}
        res = {}
        for key in info[0].keys():
            res[key] = np.stack([x[key] for x in info])
        return res

def get_carenvvec(number):
    return EnvVecs(CarEnv, number, (action_data, rd_data, 0), 2)


