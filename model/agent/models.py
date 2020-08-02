import torch
from torch import nn
import numpy as np

class CancelProbModel(nn.Module):
    def __init__(self, hourfeat = 8):
        super(CancelProbModel, self).__init__()
        self.houremb = nn.Embedding(24, hourfeat)
        
        self.fc = nn.Sequential(
            nn.Linear(hourfeat + 6, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
    # startPos, endPos, hour, reward, ETA, prob
    def forward(self, startPos, endPos, hour, reward, ETA):
        startPos = startPos.float()
        endPos = endPos.float()
        hour = self.houremb(hour)
        reward = reward.unsqueeze(1)
        ETA = ETA.unsqueeze(1)
        #print(startGID.shape, endGID.shape, hour.shape, week.shape)
        x = torch.cat((startPos, endPos, hour, reward, ETA), dim = 1)
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        return x
    
class GridModel(nn.Module):
    def __init__(self, hourfeat = 8):
        super(GridModel, self).__init__()
        self.houremb = nn.Embedding(24, hourfeat)
        
        self.order = nn.Sequential(
            nn.Linear(hourfeat + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.reward = nn.Sequential(
            nn.Linear(hourfeat + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, GPS, hour):
        GPS = GPS.float()
        hour = self.houremb(hour)
        #print(GID.shape, hour.shape, week.shape)
        x = torch.cat((GPS, hour), dim = 1)
        #print(x.shape)
        return self.order(x).squeeze(), self.reward(x).squeeze()

class DVNNet(nn.Module):
    """fetch value function
    
    Args: 
        GIDnum: number of grids
        GIDfeat: grid ID embedding feature number
        hourfeat: hour embedding feature number
        weekfeat: week embedding feature number
    """
    def __init__(self, hourfeat = 8):
        super(DVNNet, self).__init__()
        self.houremb = nn.Embedding(24, hourfeat)
        
        self.fc = nn.Sequential(
            nn.Linear(hourfeat + 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    """
    Args:
        lat: driver position lat
        lon: driver position lon
        hour: now hour
        average_reward: average reward in this grid
        demand: expected demand in this grid
    """
    def forward(self, lat, lon, hour, average_reward, demand):
        #print(lat, lon, hour, average_reward, demand)
        lat = lat.unsqueeze(1).float()
        lon = lon.unsqueeze(1).float()
        hour = self.houremb(hour)
        average_reward = average_reward.float().unsqueeze(1)
        demand = demand.float().unsqueeze(1)
        x = torch.cat((lat, lon, hour, average_reward, demand), dim = 1)
        #print(x.shape)
        return self.fc(x).squeeze()