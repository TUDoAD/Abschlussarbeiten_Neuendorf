# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 09:57:28 2022

@author: marvi
"""
import numpy as np
import pickle
from importlib.machinery import SourceFileLoader
import tensorflow as tf
import copy
#%%
# import working models

DQN = tf.keras.models.load_model(r'C:/Users/marvi/sciebo/Masterarbeit/Model/DQN2022_04_25 13_54.h5')

PPO =  tf.keras.models.load_model(r'C:/Users/marvi/sciebo/Masterarbeit/Model/PPO2022_03_17 16_43.h5')
#PPO = torch.load(r"C:/Users/marvi/PPO_test/policy.pth")
DDPG =  tf.keras.models.load_model(r'C:/Users/marvi/sciebo/Masterarbeit/Model/DDPG_optimised_net2022_04_04 16_07.h5')
#DDPG =  tf.keras.models.load_model(r'C:/Users/marvi/DDPG_model_02_03_22_1.h5')

with open(r'C:/Users/marvi/sciebo/Masterarbeit/Model/Q_table_new_env2022_04_24 09_58.pkl', 'rb') as f:
            Q_table = pickle.load(f)
#%%
DDPG.compile
#%% 
episodes = 1
wanted_size = 1.2

theta = SourceFileLoader("Theta.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/Theta.py").load_module()
reward_weights = [1,0.3,0.3,0]
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Python Module/Data_new_aceton_18_03.pkl', 'rb') as f: 
    data = pickle.load(f)
model_param = SourceFileLoader("new_model_param.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/new_model_param.py").load_module()
dis_env_load = SourceFileLoader("Environment_dis_with_aceton_25_03_22.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/Environment_dis_with_aceton_25_03_22.py").load_module()
dis_env_load = SourceFileLoader("Environment_dis_with_aceton_29_03_22.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/Environment_dis_with_aceton_29_03_22.py").load_module()

con_env_load = SourceFileLoader("Environment_con_with_aceton_28_03_22.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/Environment_con_with_aceton_28_03_22.py").load_module()
data_2, dummie = model_param.model_parameter(data['drop_size'], data['rpm'], data['solvent'], data['flooding'], data['aceton_conc'])

all_theta = {}
for key in data_2.keys():
    #print(key)
    all_theta[key]=theta.Theta(data_2,1,'rpm','drop_size', key)
#dis_env = dis_env_load.ExtractionEnv_dis(wanted_size, all_theta, data_2, reward_weights)
dis_env = dis_env_load.ExtractionEnv_dis(wanted_size, all_theta, data_2, reward_weights,10)

con_env = con_env_load.ExtractionEnv_con(wanted_size, all_theta, data_2, reward_weights)
#%%DQN

for episode in range(1, episodes+1):
    teststate = dis_env.set_reset(550,20)
    done = False
    score = 0
    DQNdrop = []
    DQNrpm  = []
    DQNflooding = []
    DQNsolvent = []
    DQNacetone = []
    while not done:
        teststate = np.reshape(teststate, [1, -1])
        print(teststate[0])
        
        DQNdrop.append(teststate[0][0])
        DQNrpm.append(teststate[0][2])
        DQNflooding.append(teststate[0][1])
        DQNsolvent.append(teststate[0][3])
        DQNacetone.append(teststate[0][4])
        new_s = copy.copy(teststate[0])
        new_s[0]=new_s[0]/2
        new_s[2]=new_s[2]/600
        new_s[3]=new_s[3]/20
        new_s[4]=new_s[4]/2
        new_s = np.reshape(new_s, [1,-1])
        action = DQN.predict(new_s)[0]
        print(action)
        del new_s
        #print(action)
        action = np.argmax(action)
        print(action)
        teststate, reward, done, info = dis_env.step(action)
        #print(state)
        score+=reward
#%% DDPG
for episode in range(1, episodes+1):
    teststate = con_env.set_reset(450,18)
    done = False
    score = 0
    DDPGdrop = []
    DDPGrpm  = []
    DDPGflooding = []
    DDPGsolvent = []
    DDPGacetone = []
    while not done:
        teststate = np.reshape(teststate, [1, -1])
        print(teststate[0])
        DDPGdrop.append(teststate[0][0])
        DDPGrpm.append(teststate[0][2])
        DDPGflooding.append(teststate[0][1])
        DDPGsolvent.append(teststate[0][3])
        DDPGacetone.append(teststate[0][4])
        new_s = copy.copy(teststate[0])
        new_s[0]=new_s[0]/2
        new_s[2]=new_s[2]/600
        new_s[3]=new_s[3]/20
        new_s[4]=new_s[4]/2
        new_s = np.reshape(new_s, [1,-1])
        action = DDPG.predict(new_s)[0]
        action[0]*=20
        action[1]*=2
        print(action)
        del new_s
        teststate, reward, done, info = con_env.step(action)
        #print(state)
        score+=reward
        
#%%PPO
for episode in range(1, episodes+1):
    teststate = dis_env.set_reset(600,20)
    done = False
    score = 0
    PPOdrop = []
    PPOrpm  = []
    PPOflooding = []
    PPOsolvent = []
    PPOacetone = []
    
    while not done:
        teststate = np.reshape(teststate, [1, -1])
        print(teststate[0])
        
        PPOdrop.append(teststate[0][0])
        PPOrpm.append(teststate[0][2])
        PPOflooding.append(teststate[0][1])
        PPOsolvent.append(teststate[0][3])
        PPOacetone.append(teststate[0][4])
        new_s = copy.copy(teststate[0])
        new_s[0]=new_s[0]/2
        new_s[2]=new_s[2]/600
        new_s[3]=new_s[3]/35
        new_s = np.reshape(new_s, [1,-1])
        action = PPO.predict(new_s)[0]
        #action[0][0]*=20
        #action[0][1]*=2
        del new_s
        print(action)
        action = np.argmax(action)
        print(action)
        teststate, reward, done, info = dis_env.step(action)
        #print(state)
        score+=reward
        
#%%Q_table
dis_env = dis_env_load.ExtractionEnv_dis(wanted_size, all_theta, data_2, reward_weights,10)
for episode in range(1, episodes+1):
    teststate = dis_env.set_reset(550,20)
    done = False
    score = 0
    Q_tabledrop = []
    Q_tablerpm  = []
    Q_tableflooding = []
    Q_tablesolvent = []
    Q_tableacetone = []
    while not done:
        teststate = np.reshape(teststate, [1, -1])
        print(teststate[0])
        
        Q_tabledrop.append(teststate[0][0])
        Q_tablerpm.append(teststate[0][2])
        Q_tableflooding.append(teststate[0][1])
        Q_tablesolvent.append(teststate[0][3])
        Q_tableacetone.append(teststate[0][4])
        action_df = Q_table.loc[(Q_table[4]==np.round(teststate[0][4],2))]#and(Q_table[3]==teststate[0][3])]
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
        teststate, reward, done, info = dis_env.step(action)
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