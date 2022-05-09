# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:45:40 2022

@author: marvi
"""

import pickle
#%%

with open(r'C:/Users/marvi/sciebo/Masterarbeit/Test DQN plus DDPG an Kolonne/extr_course_DDPG_02_05_22.pkl_start_550_35', 'rb') as f: 
    extr_course = pickle.load(f)
    
#%%
print( extr_course[0]['drop_size'])
#%%

drop = []
for item in extr_course:
    drop.append(item['state'])