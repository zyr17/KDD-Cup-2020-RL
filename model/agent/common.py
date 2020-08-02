import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
import time
import torch
import os

last_use_cuda = True

def cuda(tensor, use_cuda = None):
    """
    A cuda wrapper
    """
    global last_use_cuda
    if use_cuda == None:
        use_cuda = last_use_cuda
    last_use_cuda = use_cuda
    if not use_cuda:
        return tensor
    if tensor is None:
        return None
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor

folder = os.path.dirname(os.path.abspath(__file__)) + '/'
min_gps = np.array([104.042102,30.652828])
max_gps = np.array([104.129591,30.727818])
real_distance = np.array([8350, 8350])
min_gps = np.array([103.8,30.45])
max_gps = np.array([104.3,30.9])
real_distance = np.array([47720.28483581, 50106.68089079])
block_number = np.array([50, 50])

grid_data = pickle.load(open(folder + 'hex_grid.pkl', 'rb'))

def isRayIntersectsSegment(poi,s_poi,e_poi): #[x,y] [lng,lat]
    #输入：判断点，边起点，边终点，都是[lng,lat]格式数组
    if s_poi[1]==e_poi[1]: #排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if s_poi[1]>poi[1] and e_poi[1]>poi[1]: #线段在射线上边
        return False
    if s_poi[1]<poi[1] and e_poi[1]<poi[1]: #线段在射线下边
        return False
    if s_poi[1]==poi[1] and e_poi[1]>poi[1]: #交点为下端点，对应spoint
        return False
    if e_poi[1]==poi[1] and s_poi[1]>poi[1]: #交点为下端点，对应epoint
        return False
    if s_poi[0]<poi[0] and e_poi[0]<poi[0]: #线段在射线左边
        return False

    xseg=e_poi[0]-(e_poi[0]-s_poi[0])*(e_poi[1]-poi[1])/(e_poi[1]-s_poi[1]) #求交
    if xseg<poi[0]: #交点在射线起点的左侧
        return False
    return True  #排除上述情况之后

def isPoiWithinPoly(poi,poly):
    sinsc=0 #交点个数
    poly = [*poly, poly[0]]
    for s_poi, e_poi in zip(poly[:-1], poly[1:]): #[0,len-1]
        if isRayIntersectsSegment(poi,s_poi,e_poi):
            sinsc+=1 #有交点就加1

    return True if sinsc%2==1 else  False

def find_grid_idx(point, grid = grid_data):
    res = -1
    for i in range(len(grid['ID'])):
        if isPoiWithinPoly(point, grid['vertex'][i]):
            if res != -1:
                print(res, i, point)
            res = i
    return res

def extract_time_feat(ts):
    lt = time.localtime(ts)
    return lt.tm_mday, lt.tm_hour, lt.tm_wday, lt.tm_wday >= 5