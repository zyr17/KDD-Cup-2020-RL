#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import time
from models import DVNNet, CancelProbModel, GridModel
import numpy as np
import pickle
import KM as KM_algo
KM_algo = KM_algo.Munkres().compute
from common import find_grid_idx, extract_time_feat, min_gps, max_gps, real_distance, block_number, cuda
delta_gps = max_gps - min_gps
blocks = block_number[0] * block_number[1]


# In[2]:


GAMMA = 0.95
TICK = 200


# In[3]:


modelfolder = os.path.dirname(os.path.abspath(__file__)) + '/'
dvnnet = DVNNet()
dvnnet.load_state_dict(torch.load(modelfolder + 'DVNNet.pt')['model'])
cancelProbModel = CancelProbModel()
cancelProbModel.load_state_dict(torch.load(modelfolder + 'cancelProbModel.pt')['model'])
gridModel = GridModel()
gridModel.load_state_dict(torch.load(modelfolder + 'gridModel.pt')['model'])


# In[4]:


pkl_folder = os.path.dirname(os.path.abspath(__file__)) + '/'
rd_data = pickle.load(open(pkl_folder + 'grid_order_reward.pkl', 'rb'))
meanstd = {'order': [1.3818544802263453, 2.0466071372530115], 'reward': [0.003739948797879627, 0.000964668315987685]}
for i in meanstd.keys():
    n = meanstd[i]
    if i == 'order':
        rd_data[i] = np.log(rd_data[i] + 1)
    r = rd_data[i]
    r -= n[0]
    r /= n[1]


# In[5]:


def meanstd(arr, mean, std):
    return (arr.astype(float) - mean) / std
cancel_eta = [1327.0779045105526, 847.6405218280669]
cancel_reward = [4.182467829036552, 2.826104770240745]


# In[6]:


def obs2value(obs):
    with torch.no_grad():
        f_obs = {}
        for key in obs[0].keys():
            f_obs[key] = np.stack([np.array(x[key]) for x in obs])
        
        hour = torch.tensor([time.localtime(x).tm_hour for x in f_obs['timestamp']])
        dist_pos = (f_obs['order_driver_distance'] / 200).astype(int)
        dist_pos[dist_pos > 9] = 9
        startpos = meanstd(f_obs['order_start_location'], min_gps, delta_gps)
        endpos = meanstd(f_obs['order_finish_location'], min_gps, delta_gps)
        dist = ((np.abs(endpos - startpos) * real_distance) ** 2).sum(axis=1) ** 0.5
        eta = dist / 3 # use order_finish_timestamp ?
        args = [startpos, endpos, hour, meanstd(f_obs['reward_units'], *cancel_reward).astype('float32'), meanstd(eta, *cancel_eta).astype('float32')]
        args = [torch.tensor(x) for x in args]
        cancel = cancelProbModel(*args).numpy()
        cancel = np.choose(dist_pos, cancel.T)
        #print(args, dist_pos, cancel)
        order, reward = gridModel(torch.tensor(startpos), torch.tensor(hour))
        #print(startpos, hour, order, reward)
        args = [startpos[:,0], startpos[:,1], hour, order, reward]
        args = [torch.tensor(x) for x in args]
        vf = dvnnet(*args)
        #print(args, vf)
        for o, c, v, e in zip(obs, cancel, vf, eta):
            o['cancel_prob'] = c
            o['value'] = v.item()
            o['eta'] = e.item()
    return obs


# In[7]:


# [(2)pos, (,)hour] * x
def calc_v(data):
    data = list(zip(*data))
    pos, hour = [torch.tensor(x) for x in data]
    with torch.no_grad():
        order, reward = gridModel(pos, hour)
        vf = dvnnet(pos[:,0], pos[:,1], hour, order, reward)
        return vf


# In[8]:


def KM(obs):
    driver_id = []
    order_id = []
    driver_data = []
    dset = set()
    oset = set()
    for o in obs:
        did = o['driver_id']
        if did not in dset:
            dset.add(did)
            driver_id.append(did)
            order_id.append(-1)
            driver_data.append([o['order_start_location'], time.localtime(o['timestamp']).tm_hour])
    for o in obs:
        oid = o['order_id']
        if oid not in oset:
            oset.add(oid)
            order_id.append(oid)
    driver_v = calc_v(driver_data)
    edge = np.zeros((len(driver_id), len(order_id)), dtype='float')
    edge[:] = -1000
    for i in range(len(driver_id)):
        edge[i][i] = driver_v[i] * GAMMA
    for o in obs:
        di = driver_id.index(o['driver_id'])
        oi = order_id.index(o['order_id'])
        t = o['eta'] + o['pick_up_eta']
        t = int(t / TICK) + 1
        gt = GAMMA ** t
        p = o['cancel_prob']
        edge[di][oi] = (o['reward_units'] * (1 - gt) / (1 - GAMMA) + gt * o['value']) * (1 - p) + p * driver_v[di] * GAMMA
    res = []
    for di, oi in KM_algo(-edge):
        if order_id[oi] != -1:
            res.append({'order_id': order_id[oi], 'driver_id': driver_id[di]})
    return res


# In[9]:


def geodis(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

class Agent(object):
    """ Agent for dispatching and reposition """

    def __init__(self):
        """ Load your trained model and initialize the parameters """
        pass

    def dispatch(self, dispatch_observ):
        """ Compute the assignment between drivers and passengers at each time step
        :param dispatch_observ: a list of dict, the key in the dict includes:
            order_id, int
            driver_id, int
            order_driver_distance, float
            order_start_location, a list as [lng, lat], float
            order_finish_location, a list as [lng, lat], float
            driver_location, a list as [lng, lat], float
            timestamp, int
            order_finish_timestamp, int
            day_of_week, int
            reward_units, float
            pick_up_eta, float

        :return: a list of dict, the key in the dict includes:
            order_id and driver_id, the pair indicating the assignment
        """
        res = KM(obs2value(dispatch_observ))
        return res

    def reposition(self, repo_observ):
        """ Compute the reposition action for the given drivers
        :param repo_observ: a dict, the key in the dict includes:
            timestamp: int
            driver_info: a list of dict, the key in the dict includes:
                driver_id: driver_id of the idle driver in the treatment group, int
                grid_id: id of the grid the driver is located at, str
            day_of_week: int

        :return: a list of dict, the key in the dict includes:
            driver_id: corresponding to the driver_id in the od_list
            destination: id of the grid the driver is repositioned to, str
        """
        repo_action = []
        for driver in repo_observ['driver_info']:
            # the default reposition is to let drivers stay where they are
            repo_action.append({'driver_id': driver['driver_id'], 'destination': driver['grid_id']})
        return repo_action


# In[10]:


if __name__ == '__main__':
    sampledata = [{"order_id": 0, "driver_id": 36, "order_driver_distance": 1126.2238477454885, "order_start_location": [104.17213000000001, 30.65868], "order_finish_location": [104.07704, 30.68109], "driver_location": [104.17223539384607, 30.6485633653061], "timestamp": 1488330000, "order_finish_timestamp": 1488335000, "day_of_week": 2, "reward_units": 2.620967741935484, "pick_up_eta": 375.40794924849615}, {"order_id": 0, "driver_id": 208, "order_driver_distance": 1053.7547898479709, "order_start_location": [104.17213000000001, 30.65868], "order_finish_location": [104.07704, 30.68109], "driver_location": [104.1622195095486, 30.662794596354168], "timestamp": 1488330000, "order_finish_timestamp": 1488335000, "day_of_week": 2, "reward_units": 2.620967741935484, "pick_up_eta": 351.2515966159903}, {"order_id": 0, "driver_id": 1015, "order_driver_distance": 1138.6571871299093, "order_start_location": [104.17213000000001, 30.65868], "order_finish_location": [104.07704, 30.68109], "driver_location": [104.160415, 30.656928], "timestamp": 1488330000, "order_finish_timestamp": 1488335000, "day_of_week": 2, "reward_units": 2.620967741935484, "pick_up_eta": 379.55239570996974}, {"order_id": 0, "driver_id": 1244, "order_driver_distance": 1935.2636459261153, "order_start_location": [104.17213000000001, 30.65868], "order_finish_location": [104.07704, 30.68109], "driver_location": [104.16251256808364, 30.673970411552567], "timestamp": 1488330000, "order_finish_timestamp": 1488335000, "day_of_week": 2, "reward_units": 2.620967741935484, "pick_up_eta": 645.0878819753717}, {"order_id": 0, "driver_id": 1758, "order_driver_distance": 1324.83184991022, "order_start_location": [104.17213000000001, 30.65868], "order_finish_location": [104.07704, 30.68109], "driver_location": [104.18570855034723, 30.66096164279514], "timestamp": 1488330000, "order_finish_timestamp": 1488335000, "day_of_week": 2, "reward_units": 2.620967741935484, "pick_up_eta": 441.61061663673996}, {"order_id": 0, "driver_id": 4285, "order_driver_distance": 969.4801721971353, "order_start_location": [104.17213000000001, 30.65868], "order_finish_location": [104.07704, 30.68109], "driver_location": [104.18204833984375, 30.65693332248264], "timestamp": 1488330000, "order_finish_timestamp": 1488335000, "day_of_week": 2, "reward_units": 2.620967741935484, "pick_up_eta": 323.1600573990451}, {"order_id": 0, "driver_id": 6133, "order_driver_distance": 1228.5477420121815, "order_start_location": [104.17213000000001, 30.65868], "order_finish_location": [104.07704, 30.68109], "driver_location": [104.17724745008681, 30.64855984157986], "timestamp": 1488330000, "order_finish_timestamp": 1488335000, "day_of_week": 2, "reward_units": 2.620967741935484, "pick_up_eta": 409.5159140040605}, {"order_id": 0, "driver_id": 6844, "order_driver_distance": 654.4543412270853, "order_start_location": [104.17213000000001, 30.65868], "order_finish_location": [104.07704, 30.68109], "driver_location": [104.17593375824893, 30.66356441259522], "timestamp": 1488330000, "order_finish_timestamp": 1488335000, "day_of_week": 2, "reward_units": 2.620967741935484, "pick_up_eta": 218.15144707569507}, {"order_id": 0, "driver_id": 7510, "order_driver_distance": 726.0510402552777, "order_start_location": [104.17213000000001, 30.65868], "order_finish_location": [104.07704, 30.68109], "driver_location": [104.17337917751736, 30.652246907552083], "timestamp": 1488330000, "order_finish_timestamp": 1488335000, "day_of_week": 2, "reward_units": 2.620967741935484, "pick_up_eta": 242.0170134184259}, {"order_id": 0, "driver_id": 7700, "order_driver_distance": 821.2972562779313, "order_start_location": [104.17213000000001, 30.65868], "order_finish_location": [104.07704, 30.68109], "driver_location": [104.17636528862847, 30.652264539930556], "timestamp": 1488330000, "order_finish_timestamp": 1488335000, "day_of_week": 2, "reward_units": 2.620967741935484, "pick_up_eta": 273.76575209264377}, {"order_id": 0, "driver_id": 7899, "order_driver_distance": 825.2737189392166, "order_start_location": [104.17213000000001, 30.65868], "order_finish_location": [104.07704, 30.68109], "driver_location": [104.16680640835654, 30.65285006509489], "timestamp": 1488330000, "order_finish_timestamp": 1488335000, "day_of_week": 2, "reward_units": 2.620967741935484, "pick_up_eta": 275.09123964640554}, {"order_id": 1, "driver_id": 208, "order_driver_distance": 1439.1476028728016, "order_start_location": [104.17413, 30.67068], "order_finish_location": [104.06004, 30.66109], "driver_location": [104.1622195095486, 30.662794596354168], "timestamp": 1488330000, "order_finish_timestamp": 1488335300, "day_of_week": 2, "reward_units": 3.4274193548387095, "pick_up_eta": 479.7158676242672}, {"order_id": 1, "driver_id": 708, "order_driver_distance": 1817.8173728839545, "order_start_location": [104.17413, 30.67068], "order_finish_location": [104.06004, 30.66109], "driver_location": [104.17997381743818, 30.686217053660336], "timestamp": 1488330000, "order_finish_timestamp": 1488335300, "day_of_week": 2, "reward_units": 3.4274193548387095, "pick_up_eta": 605.9391242946515}, {"order_id": 1, "driver_id": 1244, "order_driver_distance": 1171.0770650337058, "order_start_location": [104.17413, 30.67068], "order_finish_location": [104.06004, 30.66109], "driver_location": [104.16251256808364, 30.673970411552567], "timestamp": 1488330000, "order_finish_timestamp": 1488335300, "day_of_week": 2, "reward_units": 3.4274193548387095, "pick_up_eta": 390.3590216779019}, {"order_id": 1, "driver_id": 1310, "order_driver_distance": 1874.2759095443903, "order_start_location": [104.17413, 30.67068], "order_finish_location": [104.06004, 30.66109], "driver_location": [104.16734130859375, 30.68647216796875], "timestamp": 1488330000, "order_finish_timestamp": 1488335300, "day_of_week": 2, "reward_units": 3.4274193548387095, "pick_up_eta": 624.7586365147968}, {"order_id": 1, "driver_id": 1684, "order_driver_distance": 1895.4101246531948, "order_start_location": [104.17413, 30.67068], "order_finish_location": [104.06004, 30.66109], "driver_location": [104.16728949012791, 30.68665808893226], "timestamp": 1488330000, "order_finish_timestamp": 1488335300, "day_of_week": 2, "reward_units": 3.4274193548387095, "pick_up_eta": 631.8033748843983}, {"order_id": 1, "driver_id": 1758, "order_driver_distance": 1549.0436728869656, "order_start_location": [104.17413, 30.67068], "order_finish_location": [104.06004, 30.66109], "driver_location": [104.18570855034723, 30.66096164279514], "timestamp": 1488330000, "order_finish_timestamp": 1488335300, "day_of_week": 2, "reward_units": 3.4274193548387095, "pick_up_eta": 516.3478909623219}, {"order_id": 1, "driver_id": 2065, "order_driver_distance": 1725.152898021458, "order_start_location": [104.17413, 30.67068], "order_finish_location": [104.06004, 30.66109], "driver_location": [104.17261962890625, 30.68612277560764], "timestamp": 1488330000, "order_finish_timestamp": 1488335300, "day_of_week": 2, "reward_units": 3.4274193548387095, "pick_up_eta": 575.0509660071526}, {"order_id": 1, "driver_id": 2738, "order_driver_distance": 1731.5226281126777, "order_start_location": [104.17413, 30.67068], "order_finish_location": [104.06004, 30.66109], "driver_location": [104.1697714673517, 30.685776089926435], "timestamp": 1488330000, "order_finish_timestamp": 1488335300, "day_of_week": 2, "reward_units": 3.4274193548387095, "pick_up_eta": 577.1742093708925}, {"order_id": 1, "driver_id": 4178, "order_driver_distance": 1568.392336314236, "order_start_location": [104.17413, 30.67068], "order_finish_location": [104.06004, 30.66109], "driver_location": [104.16472561451413, 30.682216094561085], "timestamp": 1488330000, "order_finish_timestamp": 1488335300, "day_of_week": 2, "reward_units": 3.4274193548387095, "pick_up_eta": 522.7974454380786}, {"order_id": 1, "driver_id": 4226, "order_driver_distance": 1904.7210297829317, "order_start_location": [104.17413, 30.67068], "order_finish_location": [104.06004, 30.66109], "driver_location": [104.16724962643582, 30.68673459605026], "timestamp": 1488330000, "order_finish_timestamp": 1488335300, "day_of_week": 2, "reward_units": 3.4274193548387095, "pick_up_eta": 634.9070099276439}, {"order_id": 1, "driver_id": 4285, "order_driver_distance": 1707.8128660040907, "order_start_location": [104.17413, 30.67068], "order_finish_location": [104.06004, 30.66109], "driver_location": [104.18204833984375, 30.65693332248264], "timestamp": 1488330000, "order_finish_timestamp": 1488335300, "day_of_week": 2, "reward_units": 3.4274193548387095, "pick_up_eta": 569.2709553346969}, {"order_id": 1, "driver_id": 6448, "order_driver_distance": 1823.2114885345427, "order_start_location": [104.17413, 30.67068], "order_finish_location": [104.06004, 30.66109], "driver_location": [104.17995406830457, 30.68627344078293], "timestamp": 1488330000, "order_finish_timestamp": 1488335300, "day_of_week": 2, "reward_units": 3.4274193548387095, "pick_up_eta": 607.7371628448476}, {"order_id": 1, "driver_id": 6670, "order_driver_distance": 1711.2011947497763, "order_start_location": [104.17413, 30.67068], "order_finish_location": [104.06004, 30.66109], "driver_location": [104.16735106195891, 30.684903441519857], "timestamp": 1488330000, "order_finish_timestamp": 1488335300, "day_of_week": 2, "reward_units": 3.4274193548387095, "pick_up_eta": 570.4003982499254}, {"order_id": 1, "driver_id": 6739, "order_driver_distance": 1716.1573286030734, "order_start_location": [104.17413, 30.67068], "order_finish_location": [104.06004, 30.66109], "driver_location": [104.18170716467857, 30.684651419281284], "timestamp": 1488330000, "order_finish_timestamp": 1488335300, "day_of_week": 2, "reward_units": 3.4274193548387095, "pick_up_eta": 572.0524428676912}, {"order_id": 1, "driver_id": 6844, "order_driver_distance": 810.7140809493975, "order_start_location": [104.17413, 30.67068], "order_finish_location": [104.06004, 30.66109], "driver_location": [104.17593375824893, 30.66356441259522], "timestamp": 1488330000, "order_finish_timestamp": 1488335300, "day_of_week": 2, "reward_units": 3.4274193548387095, "pick_up_eta": 270.2380269831325}, {"order_id": 1, "driver_id": 7581, "order_driver_distance": 1986.3777212837872, "order_start_location": [104.17413, 30.67068], "order_finish_location": [104.06004, 30.66109], "driver_location": [104.15808051215278, 30.68198784722222], "timestamp": 1488330000, "order_finish_timestamp": 1488335300, "day_of_week": 2, "reward_units": 3.4274193548387095, "pick_up_eta": 662.1259070945957}, {"order_id": 1, "driver_id": 7848, "order_driver_distance": 1531.9470260385963, "order_start_location": [104.17413, 30.67068], "order_finish_location": [104.06004, 30.66109], "driver_location": [104.16552323039143, 30.682281290577755], "timestamp": 1488330000, "order_finish_timestamp": 1488335300, "day_of_week": 2, "reward_units": 3.4274193548387095, "pick_up_eta": 510.6490086795321}]
    obs2value(sampledata)
    print(sampledata)
    a = Agent()
    print(a.dispatch(sampledata))

