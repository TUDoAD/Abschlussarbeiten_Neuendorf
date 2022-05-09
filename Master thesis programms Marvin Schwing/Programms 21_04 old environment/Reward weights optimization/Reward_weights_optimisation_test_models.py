# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 09:57:28 2022

@author: marvi
"""
import numpy as np
import pickle
from importlib.machinery import SourceFileLoader
import copy
#%%
# import working models

with open(r'C:/Users/marvi/sciebo/Masterarbeit/Programms 21_04 old environment/Reward weights optimization/Q_table models/Q_table_old_env[1, 0.3, 0.5, 0.5]2022_04_22 17_20.pkl', 'rb') as f:
            Q_table = pickle.load(f)
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
#con_env = con_env_load.ExtractionEnv_con(wanted_drop_size, all_theta, data_2, reward_weights)
        
#%%Q_table
for episode in range(1, episodes+1):
    teststate = own_env_dis.set_reset(550,35)
    done = False
    score = 0
    Q_tabledrop = []
    Q_tablerpm  = []
    Q_tableflooding = []
    Q_tablesolvent = []
   # Q_tableacetone = []
    while not done:
        teststate = np.reshape(teststate, [1, -1])
        print(teststate[0])
        
        Q_tabledrop.append(teststate[0][0])
        Q_tablerpm.append(teststate[0][2])
        Q_tableflooding.append(teststate[0][1])
        Q_tablesolvent.append(teststate[0][3])
       # Q_tableacetone.append(teststate[0][4])
        action_df = Q_table.loc[(Q_table[0]==np.round(teststate[0][0],2))]#and(Q_table[3]==teststate[0][3])]
        action_df = action_df.loc[(action_df[3]==teststate[0][3])]
        n = 10
        #print(action_df)
        #print(len(action_df))
        action_df_copy = copy.copy(action_df)
        while len(action_df)>10:
            n-=1
            #print(n)
            if n==0:
                break
            action_df = action_df_copy.loc[(action_df_copy[2]<=(teststate[0][2]+10*n))]
            action_df = action_df.loc[(action_df[2]>=(teststate[0][2]-10*n))]
            #print(len(action_df))
            #&(action_df_copy[2]>=(teststate[0][2]-10*n)) ]
            
            
        #action_df = action_df.iloc[((action_df[3]-teststate[0][3]).abs().argsort()[:])]
        #print(len(action_df))
        #print(action_df)
        #break
        #----verbesserung aus Test Q-value Table 24_01_22
        action_vektor = np.zeros(16)
        normalization_vektor = np.ones(16)
        for item in zip(action_df['Q_value'],action_df[2]):
            #print('Q-value/rpm: {}'.format(item))
            if np.count_nonzero(item[0])>0:
                Q_value_weighted = item[0]*((teststate[0][2]-np.abs(item[1]-teststate[0][2]))/teststate[0][2])
                #print('Q-value weighted: {}'.format(Q_value_weighted))
                help_normalization = np.divide(Q_value_weighted,Q_value_weighted)
                where_are_NaNs = np.isnan(help_normalization)
                help_normalization[where_are_NaNs] = 0
                normalization_vektor += help_normalization
                #print(normalization_vektor)
                action_vektor = action_vektor + Q_value_weighted
        #print('Action Matrix: {}'.format(action_vektor))
        action = np.argmax(np.divide(action_vektor,normalization_vektor))
        if max(action_vektor) == 0:
            print('Action is 5 due to an empty action vector!')
            action = 5
        
        print(action)
        teststate, reward, done, info = own_env_dis.step(action)
        #break
        #print(state)
        score+=reward
        
#%%diagramms
import matplotlib.pyplot as plt
time = np.arange(len(DQNdrop))+1
#time = np.arange(201)
fig, ax1 = plt.subplots()

ax1.set_xlabel('Action [-]',size=16)
ax1.set_ylabel('rpm [1/min]',size=16)
ax1.plot(time, DQNrpm, color='tab:red')
ax1.plot(time, DDPGrpm, color='tab:orange')
ax1.plot(time, PPOrpm, color='tab:purple')
ax1.plot(time, Q_tablerpm, color='tab:pink')
ax1.tick_params(axis='y')
ax1.axis([1, 200,300, 600 ])

#%%
fig, ax1 = plt.subplots()

ax1.set_xlabel('Action [-]',size=16)
ax1.set_ylabel('Drop size [mm]',size=16)  #already handled the x-label with ax1
ax1.plot(time, DQNdrop, color='tab:red')
ax1.plot(time, DDPGdrop, color='tab:orange')
ax1.plot(time, PPOdrop, color='tab:purple')
ax1.plot(time, Q_tabledrop, color='tab:pink')
ax1.tick_params(axis='y')
ax1.axis([1, 200, 0.5, 1.5])
#fig.tight_layout()  # otherwise the right y-label is slightly clipped
#fig.set_size_inches(25, 8)
#%%
fig, ax1 = plt.subplots()

ax1.set_xlabel('Action [-]',size=16)
ax1.set_ylabel('solvent',size=16)
ax1.plot(time, DQNsolvent, color='tab:red')
ax1.plot(time, DDPGsolvent, color='tab:orange')
ax1.plot(time, PPOsolvent, color='tab:purple')
ax1.plot(time, Q_tablesolvent, color='tab:pink')
ax1.tick_params(axis='y')
ax1.axis([1, 200,10, 20 ])

#%%
fig, ax1 = plt.subplots()

ax1.set_xlabel('Action [-]',size=16)
ax1.set_ylabel('acetone',size=16)
ax1.plot(time, DQNacetone, color='tab:red')
ax1.plot(time, DDPGacetone, color='tab:orange')
ax1.plot(time, PPOacetone, color='tab:purple')
ax1.plot(time, Q_tableacetone, color='tab:pink')
ax1.tick_params(axis='y')
ax1.axis([1, 200,0.8, 2 ])