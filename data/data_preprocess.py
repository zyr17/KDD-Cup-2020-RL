# ./raw/
# |---zip/
#     |---gps_01-10.zip
#     |---gps_11-20.zip
#     |---gps_21-30.zip
#     |---order_01-30.zip
# 
# (after unzip, about 100GB data)
# 
# ./raw/
# |---zip/
# |   |---...
# |---gps/
# |   |---gps_201611?? * 30
# |---order/
#     |---total_order_cancellation_probability/
#     |   |---order_201611??_cancel_prob * 30
#     |---total_ride_request/
#     |   |---order_201611?? * 30
#     |---hexagon_grid_table.csv
#     |---idle_transition_probability
# 
# (after preprocess)
# 
# ./
# |---raw/
# |   |---...
# |---pkl/
#     |---hex_grid.pkl
#     |---trans_prob.pkl
#     |---ride_req.pkl
#     |---cancel_prob.pkl
#     |---gps/
#         |---201611??.pkl * 30

import pickle
import os
import numpy as np
import pdb
import argparse

# require unzip
def unzip(force = False):
    if force:
        arg = '-o'
    else:
        arg = '-n'
    os.system('unzip %s -d raw/order raw/zip/order_01-30.zip' % arg)
    os.system('unzip %s -d raw/gps raw/zip/gps_01-10.zip' % arg)
    os.system('unzip %s -d raw/gps raw/zip/gps_11-20.zip' % arg)
    os.system('unzip %s -d raw/gps raw/zip/gps_21-30.zip' % arg)
    if not os.path.exists('./pkl'):
        os.mkdir('./pkl')

def hexagon_grid_table(force = False):
    filename = 'pkl/hex_grid.pkl'
    if os.path.exists(filename) and not force:
        print(filename, 'exists, skip')
        return
    lines = [x.strip().split(',') for x in open('raw/order/hexagon_grid_table.csv').readlines()]
    ID = []
    vertex = []
    for line in lines:
        if len(line) < 13:
            print('In hexagon: length mismatch, skip.', line)
            continue
        ID.append(int(line[0], 16))
        vertex.append(list(map(float, line[1:])))
    ID = np.array(ID, dtype='uint64')
    vertex = np.array(vertex).reshape(-1, 6, 2)
    vertex_x = np.array(vertex[:,:,0])
    vertex_y = np.array(vertex[:,:,1])
    vertex[:,:,0] = vertex_y
    vertex[:,:,1] = vertex_x
    #pdb.set_trace()
    print('hexagon_grid_table:', len(ID), 'records')
    pickle.dump({'ID': ID, 'vertex': vertex}, open(filename, 'wb'))

def idle_transition_probability(force = False):
    filename = 'pkl/trans_prob.pkl'
    if os.path.exists(filename) and not force:
        print(filename, 'exists, skip')
        return
    lines = [x.strip().split(',') for x in open('raw/order/idle_transition_probability').readlines()]
    time = []
    ID = []
    prob = []
    for line in lines:
        time.append(int(line[0]))
        ID.append([int(line[1], 16), int(line[2], 16)])
        prob.append(float(line[3]))
    time = np.array(time, dtype='uint8')
    ID = np.array(ID, dtype='uint64')
    prob = np.array(prob, dtype='float64')
    #pdb.set_trace()
    print('idle_transition_probability:', len(ID), 'records')
    print('    distinct src ID:', np.unique(ID[:,0]).shape[0], ', dst ID:', np.unique(ID[:,1]).shape[0])
    pickle.dump({'time': time, 'ID': ID, 'prob': prob}, open(filename, 'wb'))

def total_ride_request(force = False):
    filename = 'pkl/ride_req.pkl'
    if os.path.exists(filename) and not force:
        print(filename, 'exists, skip')
        return
    lines = []
    for i in range(1, 31):
        lines += [x.strip().split(',') for x in open('raw/order/total_ride_request/order_201611%02d' % i).readlines()]
    ID = []
    time = []
    pos = []
    reward = []
    for line in lines:
        ID.append(line[0])
        time.append([int(line[1]), int(line[2])])
        pos.append([[float(line[3]), float(line[4])], [float(line[5]), float(line[6])]])
        reward.append(float(line[7]))
    time = np.array(time)
    pos = np.array(pos)
    reward = np.array(reward)
    #pdb.set_trace()
    print('total_ride_request:', len(ID), 'records')
    pickle.dump({'ID': ID, 'time': time, 'pos': pos, 'reward': reward}, open(filename, 'wb'))

def total_order_cancellation_probability(force = False):
    filename = 'pkl/cancel_prob.pkl'
    if os.path.exists(filename) and not force:
        print(filename, 'exists, skip')
        return
    lines = []
    for i in range(1, 31):
        lines += [x.strip().split(',') for x in open('raw/order/total_order_cancellation_probability/order_201611%02d_cancel_prob' % i).readlines()]
    ID = []
    prob = []
    for line in lines:
        ID.append(line[0])
        prob.append(list(map(float, line[1:])))
    prob = np.array(prob)
    #pdb.set_trace()
    print('total_order_cancellation_probability:', len(ID), 'records')
    pickle.dump({'ID': ID, 'prob': prob}, open(filename, 'wb'))

def gps(force = False):
    foldername = 'pkl/gps/'
    filename_fmt = 'pkl/gps/201611%02d.pkl'
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    for i in range(1, 31):
        driver_ID = []
        order_ID = []
        time = []
        pos = []
        filename = filename_fmt % i
        if os.path.exists(filename) and not force:
            print(filename, 'exists, skip')
            continue
        lines = open('raw/gps/gps_201611%02d' % i).readlines()
        for num, line in enumerate(lines):
            #if num % 100000 == 0: print(num, '/', len(lines))
            line = line.strip().split(',')
            did = line[0]
            oid = line[1]
            if len(driver_ID) == 0 or not (did == driver_ID[-1] and oid == order_ID[-1]):
                if len(driver_ID):
                    time[-1] = np.array(time[-1])
                    pos[-1] = np.array(pos[-1])
                driver_ID.append(did)
                order_ID.append(oid)
                time.append([])
                pos.append([])
            time[-1].append(int(line[2]))
            pos[-1].append([float(line[3]), float(line[4])])
        time[-1] = np.array(time[-1])
        pos[-1] = np.array(pos[-1])
        #pdb.set_trace()
        print('total_gps_201611%02d:' % i, len(driver_ID), 'records')
        pickle.dump({'driver_ID': driver_ID, 'order_ID': order_ID, 'time': time, 'pos': pos}, open(filename, 'wb'))

if __name__ == "__main__":
    unzip()
    hexagon_grid_table()
    idle_transition_probability()
    total_ride_request()
    total_order_cancellation_probability()
    gps()