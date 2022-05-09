# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:23:47 2022

@author: marvi
"""

import numpy as np

def Theta(data, degree, x, y, solv):
    for t in range(degree+1):
        if t ==0:
            X = np.ones(len(data[solv][x]))
            X = np.reshape(X, (1,-1))
        if t>0:
            X = np.concatenate((X,np.reshape(np.array(data[solv][x])**t,(1,-1))))
            
    X = np.transpose(X)
    Y = data[solv][y]
    Y = np.reshape(Y,(-1,1))
    Theta = np.dot(np.linalg.inv(np.dot(np.transpose(X),X)), np.dot(np.transpose(X),np.array(Y)))
    
    return Theta