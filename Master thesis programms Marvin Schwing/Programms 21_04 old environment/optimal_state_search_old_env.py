# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:11:14 2022

@author: marvi
"""
import numpy as np
import pickle
from importlib.machinery import SourceFileLoader
#%%

episodes = 1
wanted_drop_size = 1

reward_weights = [1,0.3,0.3,0]
#load data for environment creation
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Programms 21_04 old environment/Environments/Environment data/theta_old_env.pkl', 'rb') as f: 
    theta = pickle.load(f)
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Programms 21_04 old environment/Environments/Environment data/excel_data_old_env.pkl', 'rb') as f: 
    excel_data = pickle.load(f)

#dis_env = dis_env_load.ExtractionEnv_dis(wanted_size, all_theta, data_2, reward_weights)
rpm_scale = 10
dis_env = SourceFileLoader("Environment_drop_size_dis_21_04_22.py", r"C:/Users/marvi/sciebo/Masterarbeit/Programms 21_04 old environment/Environments/Environment_drop_size_dis_21_04.py").load_module()

own_env_dis = dis_env.ExtractionEnv_dis(wanted_drop_size, theta, excel_data, reward_weights,rpm_scale)

#%%

for rpm in range(300,560,10):
    for solvent in range(25,36):
        state = own_env_dis.set_reset(rpm,solvent)
        state, reward ,done,info = own_env_dis.step(5)
        print('state: {} reward: {} done: {}'.format(state,reward,done))

#%%

dis_env = SourceFileLoader("Environment_dis_with_aceton_opt_rew_27_04_22.py", r"C:/Users/marvi/sciebo/Masterarbeit/Programms 11_04 new environment/Environments/Environment_dis_with_aceton_opt_rew_27_04_22.py").load_module()
wanted_acetone_conc=1.2
#set own reward_weights
reward_weights = [1, 0.3, 0.3, 0]
#load data for environment creation
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Programms 11_04 new environment/Environments/Environment data/theta.pkl', 'rb') as f: 
    theta = pickle.load(f)
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Programms 11_04 new environment/Environments/Environment data/excel_data.pkl', 'rb') as f: 
    excel_data = pickle.load(f)
    
own_new_env_dis = dis_env.ExtractionEnv_dis(wanted_acetone_conc, theta, excel_data,  reward_weights,10) 
    
    
#%%
highest_rew = 0
for rpm in range(300,560,10):
    for solvent in range(10,21):
        state = own_new_env_dis.set_reset(rpm,solvent)
        state, reward ,done,info = own_new_env_dis.step(5)
        if reward > highest_rew:
            highest_rew = reward
            opt_state = state
        print('state: {} reward: {} done: {}'.format(state,reward,done))
        
print('highest reward: {}'.format(highest_rew))
print('in state: {}'.format(opt_state))

        
        
        
        
        
        
        
        
        
        
        
        