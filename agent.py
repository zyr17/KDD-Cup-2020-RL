#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

FILEPATH = os.path.dirname(os.path.abspath(__file__))

import torch
nn = torch.nn
import time
#from models import DVNNet, CancelProbModel, GridModel
import numpy as np
import pickle
#import KM as KM_algo
#KM_algo = KM_algo.Munkres().compute
#from common import min_gps, max_gps, real_distance, block_number, cuda
#delta_gps = max_gps - min_gps
#blocks = block_number[0] * block_number[1]


# In[2]:


GAMMA = 0.95
TICK = 200


# In[ ]:


"""
Introduction
============
The Munkres module provides an implementation of the Munkres algorithm
(also called the Hungarian algorithm or the Kuhn-Munkres algorithm),
useful for solving the Assignment Problem.
For complete usage documentation, see: http://software.clapper.org/munkres/
"""

__docformat__ = 'markdown'

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import sys
import copy
from typing import Union, NewType, Sequence, Tuple, Optional, Callable

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__     = ['Munkres', 'make_cost_matrix', 'DISALLOWED']

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

AnyNum = NewType('AnyNum', Union[int, float])
Matrix = NewType('Matrix', Sequence[Sequence[AnyNum]])

# Info about the module
__version__   = "1.1.2"
__author__    = "Brian Clapper, bmc@clapper.org"
__url__       = "http://software.clapper.org/munkres/"
__copyright__ = "(c) 2008-2019 Brian M. Clapper"
__license__   = "Apache Software License"

# Constants
class DISALLOWED_OBJ(object):
    def __neg__(self):
        return self
DISALLOWED = DISALLOWED_OBJ()
DISALLOWED_PRINTVAL = "D"

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class UnsolvableMatrix(Exception):
    """
    Exception raised for unsolvable matrices
    """
    pass

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

class Munkres:
    """
    Calculate the Munkres solution to the classical assignment problem.
    See the module documentation for usage.
    """

    def __init__(self):
        """Create a new instance"""
        self.C = None
        self.row_covered = []
        self.col_covered = []
        self.n = 0
        self.Z0_r = 0
        self.Z0_c = 0
        self.marked = None
        self.path = None

    def pad_matrix(self, matrix: Matrix, pad_value: int=0) -> Matrix:
        """
        Pad a possibly non-square matrix to make it square.
        **Parameters**
        - `matrix` (list of lists of numbers): matrix to pad
        - `pad_value` (`int`): value to use to pad the matrix
        **Returns**
        a new, possibly padded, matrix
        """
        max_columns = 0
        total_rows = len(matrix)

        for row in matrix:
            max_columns = max(max_columns, len(row))

        total_rows = max(max_columns, total_rows)

        new_matrix = []
        for row in matrix:
            row_len = len(row)
            new_row = row[:]
            if total_rows > row_len:
                # Row too short. Pad it.
                new_row += [pad_value] * (total_rows - row_len)
            new_matrix += [new_row]

        while len(new_matrix) < total_rows:
            new_matrix += [[pad_value] * total_rows]

        return new_matrix

    def compute(self, cost_matrix: Matrix) -> Sequence[Tuple[int, int]]:
        """
        Compute the indexes for the lowest-cost pairings between rows and
        columns in the database. Returns a list of `(row, column)` tuples
        that can be used to traverse the matrix.
        **WARNING**: This code handles square and rectangular matrices. It
        does *not* handle irregular matrices.
        **Parameters**
        - `cost_matrix` (list of lists of numbers): The cost matrix. If this
          cost matrix is not square, it will be padded with zeros, via a call
          to `pad_matrix()`. (This method does *not* modify the caller's
          matrix. It operates on a copy of the matrix.)
        **Returns**
        A list of `(row, column)` tuples that describe the lowest cost path
        through the matrix
        """
        self.C = self.pad_matrix(cost_matrix)
        self.n = len(self.C)
        self.original_length = len(cost_matrix)
        self.original_width = len(cost_matrix[0])
        self.row_covered = [False for i in range(self.n)]
        self.col_covered = [False for i in range(self.n)]
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = self.__make_matrix(self.n * 2, 0)
        self.marked = self.__make_matrix(self.n, 0)

        done = False
        step = 1

        steps = { 1 : self.__step1,
                  2 : self.__step2,
                  3 : self.__step3,
                  4 : self.__step4,
                  5 : self.__step5,
                  6 : self.__step6 }

        while not done:
            try:
                func = steps[step]
                step = func()
            except KeyError:
                done = True

        # Look for the starred columns
        results = []
        for i in range(self.original_length):
            for j in range(self.original_width):
                if self.marked[i][j] == 1:
                    results += [(i, j)]

        return results

    def __copy_matrix(self, matrix: Matrix) -> Matrix:
        """Return an exact copy of the supplied matrix"""
        return copy.deepcopy(matrix)

    def __make_matrix(self, n: int, val: AnyNum) -> Matrix:
        """Create an *n*x*n* matrix, populating it with the specific value."""
        matrix = []
        for i in range(n):
            matrix += [[val for j in range(n)]]
        return matrix

    def __step1(self) -> int:
        """
        For each row of the matrix, find the smallest element and
        subtract it from every element in its row. Go to Step 2.
        """
        C = self.C
        n = self.n
        for i in range(n):
            vals = [x for x in self.C[i] if x is not DISALLOWED]
            if len(vals) == 0:
                # All values in this row are DISALLOWED. This matrix is
                # unsolvable.
                raise UnsolvableMatrix(
                    "Row {0} is entirely DISALLOWED.".format(i)
                )
            minval = min(vals)
            # Find the minimum value for this row and subtract that minimum
            # from every element in the row.
            for j in range(n):
                if self.C[i][j] is not DISALLOWED:
                    self.C[i][j] -= minval
        return 2

    def __step2(self) -> int:
        """
        Find a zero (Z) in the resulting matrix. If there is no starred
        zero in its row or column, star Z. Repeat for each element in the
        matrix. Go to Step 3.
        """
        n = self.n
        for i in range(n):
            for j in range(n):
                if (self.C[i][j] == 0) and                         (not self.col_covered[j]) and                         (not self.row_covered[i]):
                    self.marked[i][j] = 1
                    self.col_covered[j] = True
                    self.row_covered[i] = True
                    break

        self.__clear_covers()
        return 3

    def __step3(self) -> int:
        """
        Cover each column containing a starred zero. If K columns are
        covered, the starred zeros describe a complete set of unique
        assignments. In this case, Go to DONE, otherwise, Go to Step 4.
        """
        n = self.n
        count = 0
        for i in range(n):
            for j in range(n):
                if self.marked[i][j] == 1 and not self.col_covered[j]:
                    self.col_covered[j] = True
                    count += 1

        if count >= n:
            step = 7 # done
        else:
            step = 4

        return step

    def __step4(self) -> int:
        """
        Find a noncovered zero and prime it. If there is no starred zero
        in the row containing this primed zero, Go to Step 5. Otherwise,
        cover this row and uncover the column containing the starred
        zero. Continue in this manner until there are no uncovered zeros
        left. Save the smallest uncovered value and Go to Step 6.
        """
        step = 0
        done = False
        row = 0
        col = 0
        star_col = -1
        while not done:
            (row, col) = self.__find_a_zero(row, col)
            if row < 0:
                done = True
                step = 6
            else:
                self.marked[row][col] = 2
                star_col = self.__find_star_in_row(row)
                if star_col >= 0:
                    col = star_col
                    self.row_covered[row] = True
                    self.col_covered[col] = False
                else:
                    done = True
                    self.Z0_r = row
                    self.Z0_c = col
                    step = 5

        return step

    def __step5(self) -> int:
        """
        Construct a series of alternating primed and starred zeros as
        follows. Let Z0 represent the uncovered primed zero found in Step 4.
        Let Z1 denote the starred zero in the column of Z0 (if any).
        Let Z2 denote the primed zero in the row of Z1 (there will always
        be one). Continue until the series terminates at a primed zero
        that has no starred zero in its column. Unstar each starred zero
        of the series, star each primed zero of the series, erase all
        primes and uncover every line in the matrix. Return to Step 3
        """
        count = 0
        path = self.path
        path[count][0] = self.Z0_r
        path[count][1] = self.Z0_c
        done = False
        while not done:
            row = self.__find_star_in_col(path[count][1])
            if row >= 0:
                count += 1
                path[count][0] = row
                path[count][1] = path[count-1][1]
            else:
                done = True

            if not done:
                col = self.__find_prime_in_row(path[count][0])
                count += 1
                path[count][0] = path[count-1][0]
                path[count][1] = col

        self.__convert_path(path, count)
        self.__clear_covers()
        self.__erase_primes()
        return 3

    def __step6(self) -> int:
        """
        Add the value found in Step 4 to every element of each covered
        row, and subtract it from every element of each uncovered column.
        Return to Step 4 without altering any stars, primes, or covered
        lines.
        """
        minval = self.__find_smallest()
        events = 0 # track actual changes to matrix
        for i in range(self.n):
            for j in range(self.n):
                if self.C[i][j] is DISALLOWED:
                    continue
                if self.row_covered[i]:
                    self.C[i][j] += minval
                    events += 1
                if not self.col_covered[j]:
                    self.C[i][j] -= minval
                    events += 1
                if self.row_covered[i] and not self.col_covered[j]:
                    events -= 2 # change reversed, no real difference
        if (events == 0):
            raise UnsolvableMatrix("Matrix cannot be solved!")
        return 4

    def __find_smallest(self) -> AnyNum:
        """Find the smallest uncovered value in the matrix."""
        minval = sys.maxsize
        for i in range(self.n):
            for j in range(self.n):
                if (not self.row_covered[i]) and (not self.col_covered[j]):
                    if self.C[i][j] is not DISALLOWED and minval > self.C[i][j]:
                        minval = self.C[i][j]
        return minval


    def __find_a_zero(self, i0: int = 0, j0: int = 0) -> Tuple[int, int]:
        """Find the first uncovered element with value 0"""
        row = -1
        col = -1
        i = i0
        n = self.n
        done = False

        while not done:
            j = j0
            while True:
                if (self.C[i][j] == 0) and                         (not self.row_covered[i]) and                         (not self.col_covered[j]):
                    row = i
                    col = j
                    done = True
                j = (j + 1) % n
                if j == j0:
                    break
            i = (i + 1) % n
            if i == i0:
                done = True

        return (row, col)

    def __find_star_in_row(self, row: Sequence[AnyNum]) -> int:
        """
        Find the first starred element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        col = -1
        for j in range(self.n):
            if self.marked[row][j] == 1:
                col = j
                break

        return col

    def __find_star_in_col(self, col: Sequence[AnyNum]) -> int:
        """
        Find the first starred element in the specified row. Returns
        the row index, or -1 if no starred element was found.
        """
        row = -1
        for i in range(self.n):
            if self.marked[i][col] == 1:
                row = i
                break

        return row

    def __find_prime_in_row(self, row) -> int:
        """
        Find the first prime element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        col = -1
        for j in range(self.n):
            if self.marked[row][j] == 2:
                col = j
                break

        return col

    def __convert_path(self,
                       path: Sequence[Sequence[int]],
                       count: int) -> None:
        for i in range(count+1):
            if self.marked[path[i][0]][path[i][1]] == 1:
                self.marked[path[i][0]][path[i][1]] = 0
            else:
                self.marked[path[i][0]][path[i][1]] = 1

    def __clear_covers(self) -> None:
        """Clear all covered matrix cells"""
        for i in range(self.n):
            self.row_covered[i] = False
            self.col_covered[i] = False

    def __erase_primes(self) -> None:
        """Erase all prime markings"""
        for i in range(self.n):
            for j in range(self.n):
                if self.marked[i][j] == 2:
                    self.marked[i][j] = 0

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def make_cost_matrix(
        profit_matrix: Matrix,
        inversion_function: Optional[Callable[[AnyNum], AnyNum]] = None
    ) -> Matrix:
    """
    Create a cost matrix from a profit matrix by calling `inversion_function()`
    to invert each value. The inversion function must take one numeric argument
    (of any type) and return another numeric argument which is presumed to be
    the cost inverse of the original profit value. If the inversion function
    is not provided, a given cell's inverted value is calculated as
    `max(matrix) - value`.
    This is a static method. Call it like this:
        from munkres import Munkres
        cost_matrix = Munkres.make_cost_matrix(matrix, inversion_func)
    For example:
        from munkres import Munkres
        cost_matrix = Munkres.make_cost_matrix(matrix, lambda x : sys.maxsize - x)
    **Parameters**
    - `profit_matrix` (list of lists of numbers): The matrix to convert from
       profit to cost values.
    - `inversion_function` (`function`): The function to use to invert each
       entry in the profit matrix.
    **Returns**
    A new matrix representing the inversion of `profix_matrix`.
    """
    if not inversion_function:
      maximum = max(max(row) for row in profit_matrix)
      inversion_function = lambda x: maximum - x

    cost_matrix = []
    for row in profit_matrix:
        cost_matrix.append([inversion_function(value) for value in row])
    return cost_matrix

def print_matrix(matrix: Matrix, msg: Optional[str] = None) -> None:
    """
    Convenience function: Displays the contents of a matrix of integers.
    **Parameters**
    - `matrix` (list of lists of numbers): The matrix to print
    - `msg` (`str`): Optional message to print before displaying the matrix
    """
    import math

    if msg is not None:
        print(msg)

    # Calculate the appropriate format width.
    width = 0
    for row in matrix:
        for val in row:
            if val is DISALLOWED:
                val = DISALLOWED_PRINTVAL
            width = max(width, len(str(val)))

    # Make the format string
    format = ('%%%d' % width)

    # Print the matrix
    for row in matrix:
        sep = '['
        for val in row:
            if val is DISALLOWED:
                formatted = ((format + 's') % DISALLOWED_PRINTVAL)
            else: formatted = ((format + 'd') % val)
            sys.stdout.write(sep + formatted)
            sep = ', '
        sys.stdout.write(']\n')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    matrices = [
        # Square
        ([[400, 150, 400],
          [400, 450, 600],
          [300, 225, 300]],
         850),  # expected cost

        # Rectangular variant
        ([[400, 150, 400, 1],
          [400, 450, 600, 2],
          [300, 225, 300, 3]],
         452),  # expected cost


        # Square
        ([[10, 10,  8],
          [9,  8,  1],
          [9,  7,  4]],
         18),

        # Rectangular variant
        ([[10, 10,  8, 11],
          [9,  8,  1, 1],
          [9,  7,  4, 10]],
         15),

        # Rectangular with DISALLOWED
        ([[4, 5, 6, DISALLOWED],
          [1, 9, 12, 11],
          [DISALLOWED, 5, 4, DISALLOWED],
          [12, 12, 12, 10]],
         20),

        # DISALLOWED to force pairings
        ([[1, DISALLOWED, DISALLOWED, DISALLOWED],
          [DISALLOWED, 2, DISALLOWED, DISALLOWED],
          [DISALLOWED, DISALLOWED, 3, DISALLOWED],
          [DISALLOWED, DISALLOWED, DISALLOWED, 4]],
         10)]

    m = Munkres()
    for cost_matrix, expected_total in matrices:
        print_matrix(cost_matrix, msg='cost matrix')
        indexes = m.compute(-np.array(cost_matrix))
        total_cost = 0
        for r, c in indexes:
            x = cost_matrix[r][c]
            total_cost += x
            print(('(%d, %d) -> %d' % (r, c, x)))
        print(('lowest cost=%d' % total_cost))
        #assert expected_total == total_cost
        
KM_algo = Munkres().compute


# In[ ]:


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

folder = FILEPATH + '/'
min_gps = np.array([104.042102,30.652828])
max_gps = np.array([104.129591,30.727818])
real_distance = np.array([8350, 8350])
min_gps = np.array([103.8,30.45])
max_gps = np.array([104.3,30.9])
real_distance = np.array([47720.28483581, 50106.68089079])
block_number = np.array([50, 50])
delta_gps = max_gps - min_gps
blocks = block_number[0] * block_number[1]


# In[ ]:


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


# In[3]:


modelfolder = FILEPATH + '/'
dvnnet = DVNNet()
dvnnet.load_state_dict(torch.load(modelfolder + 'DVNNet.pt')['model'])
cancelProbModel = CancelProbModel()
cancelProbModel.load_state_dict(torch.load(modelfolder + 'cancelProbModel.pt')['model'])
gridModel = GridModel()
gridModel.load_state_dict(torch.load(modelfolder + 'gridModel.pt')['model'])


# In[4]:


pkl_folder = FILEPATH + '/'
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

