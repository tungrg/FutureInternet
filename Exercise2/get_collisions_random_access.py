#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:38:41 2024

@author: frank
"""

try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import pandas as pd
import numpy as np

#number_transmissions defines the number of uniform random transmissions that are done within
#0 seconds and 'sim_end' seconds. The tranasmission_duration can be a fixed value or an array
# of length 'number_transmissions'
def get_collisions_random(number_transmissions, sim_end, transmission_duration):
    
    #get transmission start times
    t_start = np.random.uniform(0,sim_end,number_transmissions)
    #and ends
    t_end = t_start+transmission_duration
    collisionArray = np.zeros(len(t_start))
    
    #get all intervals
    all_intervals = pd.IntervalIndex.from_arrays(t_start, t_end, closed='both')
    #iterate through array
    for i in range(len(t_start)):
        overlap_list = all_intervals.overlaps(pd.Interval(t_start[i], t_end[i]))
        if np.sum(overlap_list) > 1:
            collisionArray[i] = 1
        else:
            collisionArray[i] = 0
    return collisionArray