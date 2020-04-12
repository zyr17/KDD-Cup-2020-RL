# -*- coding: utf-8 -*-
# @File: agent.py
# @Author: Xiaocheng Tang
# @Date:   2020-03-17 17:03:34

import random

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
    for od_info in dispatch_observ:
      od_info['go_dis'] = geodis(od_info['order_start_location'], od_info['driver_location'])
      od_info['order_dis'] = geodis(od_info['order_start_location'], od_info['order_finish_location'])
    #dispatch_observ.sort(key=lambda od_info: od_info['reward_units'] * 1e6 - od_info['pick_up_eta'], reverse=True)
    #dispatch_observ.sort(key=lambda od_info: od_info['reward_units'] / (od_info['go_dis'] + 1e-8), reverse=True)
    random.shuffle(dispatch_observ)
    assigned_order = set()
    assigned_driver = set()
    dispatch_action = []
    for od in dispatch_observ:
      # make sure each order is assigned to one driver, and each driver is assigned with one order
      if (od["order_id"] in assigned_order) or (od["driver_id"] in assigned_driver):
        continue
      assigned_order.add(od["order_id"])
      assigned_driver.add(od["driver_id"])
      dispatch_action.append(dict(order_id=od["order_id"], driver_id=od["driver_id"]))
    return dispatch_action

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
