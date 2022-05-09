# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 11:10:33 2022

@author: marvi
"""
import numpy as np

def model_parameter(drop, rpm, solvent, flood):
    sliced_data = {}
    params = {}
    t = 0
    r = []
    d = []
    f = []
    for solv in solvent:
        r.append(rpm[t])
        d.append(drop[t]) 
        f.append(flood[t])
        if (t+1)>=len(solvent):
            sliced_data['solvent {}'.format(solvent[t])] = {'rpm':r, 'drop_size':d, 'flooding':f}
            
            #---calculate omega---
            ones = np.ones(len(r))
            #np.transpose(np.array(x_drop_size))
            X = ones,np.array(r)#,np.array(r)**2,np.array(r)**3
            X = np.transpose(X)
            Omega = np.dot(np.linalg.inv(np.dot(np.transpose(X),X)), np.dot(np.transpose(X),np.transpose(np.array(d))))
            #---calculate omega ends---
            
            params['solvent {}'.format(solvent[t])] = Omega
            r = []
            d = []
            f = []
        elif (solvent[t+1]>solvent[t]):
            sliced_data['solvent {}'.format(solvent[t])] = {'rpm':r, 'drop_size':d, 'flooding':f}
            
            #---calculate omega---
            ones = np.ones(len(r))
            #np.transpose(np.array(x_drop_size))
            X = ones,np.array(r)#,np.array(r)**2,np.array(r)**3
            X = np.transpose(X)
            Omega = np.dot(np.linalg.inv(np.dot(np.transpose(X),X)), np.dot(np.transpose(X),np.transpose(np.array(d))))
            #---calculate omega ends---
            
            params['solvent {}'.format(solvent[t])] = Omega
            
            r = []
            d = []
            f = []
        t+=1
        
        
        
    return sliced_data, params