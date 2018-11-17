#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:34:42 2017

@author: Dascienz
"""

import numpy as np

individual = np.array([5.,0.,2.,0.])
arrays = np.array([[5.,0.,2.,0.],
                  [1.,1.,1.,1.],
                  [5.,0.,1.,0.]])

def cosDist(x,y):
        num = np.dot(x,y)
        den = np.sqrt(np.dot(x,x)*np.dot(y,y))
        return (1.-(num/den))

def cosSim(individual, arrays):
    N = len(arrays)
    distances = np.zeros(N)
    for idx in range(N):
        distances[idx] = cosDist(individual, arrays[idx])
    return distances      