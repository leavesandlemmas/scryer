# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 11:51:58 2022

@author: fso
"""

import numpy as np
from scipy.spatial import KDTree

Mx = KDTree(x)
My = KDTree(y)

def crossmap(Mx,My):
    dist, simplices = Mx.query(Mx.data, Mx.m + 2)
    u = np.exp(-dist[:,1:]/dist[:,1,None])
    w = u/np.sum(u, axis=1)[:,None]
    ywt = w[...,None] * My.data[simplices[:,1:]]
    ypred = ywt.sum(axis=1)
    return ypred
 
    
    