# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 09:57:28 2022

@author: marvi
"""

from gym import Env
#from gym.spaces import Discrete, Box
#from gym import spaces
import gym
import numpy as np
import pickle
from importlib.machinery import SourceFileLoader
import random
import tensorflow as tf
#import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo.policies import MlpPolicy
import copy
#import torch
#from stable_baselines3 import PPO
#environment with discrete actions
class ExtractionEnv_dis(Env):
    #reward_weights = [drop_size_weight, rpm_weight, solvent_weight, action_weight]
    def __init__(self, wanted_size, omega, data, reward_weights):
        
        #parameter for reward calculation
        #self.flooding_time = 0
        #action space in diskret umbauen!
        self.action_space = gym.spaces.Discrete(16)
        #self.action_space = spaces.Box(low = np.array([-20,-2]), high = np.array([20,+2]), shape=(2,), dtype=np.float32)                                        
                                                    #[drop,flooding,rpm,solvent]
        self.observation_space = gym.spaces.Box(low = np.array([0,0,200,25]), high = np.array([2,1,600,35], dtype=np.float32))
        self.data = data
        self.omega = omega
        self.wanted_size = wanted_size
        self.reward_weights = reward_weights
        # Set start 
        # randomize the dropsize
        # calculation of the rpm in dependence of the drop size
        # Set Extraction Length
        self.extract_length = 200
    def step(self, action):
        # Apply action: Change rpm or solvent flow
        # 28.02.22: Change of the reward: should be between -1 and 1
        reward = 0
        done= False
        if action < 11:
            self.state[2] += (action-5)*2 #kleinere änderung beim nächsten mal!!!!!
        if action >= 11:                    #25.02 rpm schritte verkleinert
            self.state[3] += (action-13)
            
        if (self.state[3] < 25) or (self.state[3] > 35):
            done = True
            reward = -1
            #reward = -50 - self.extract_length
            if self.state[3] < 25:
                Omega = self.omega['solvent 25']
                self.state[3] = 25
            else:
                Omega = self.omega['solvent 35']
                self.state[3] = 35
        else:
            Omega = self.omega['solvent {}'.format(self.state[3])]

        
        # update other parameters
        drop_omega = 0
        for t in range(len(Omega)):
            drop_omega = drop_omega + Omega[t]*self.state[2]**t
        
        self.state[0] = round(drop_omega[0] + random.uniform(-0.10, 0.10) ,3)
        
        self.state[1] = round(self.data['solvent {}'.format(self.state[3])]['flooding'][self.data['solvent {}'.format(self.state[3])]['rpm'].index(round(self.state[2]/50)*50)])
        
        
        # Reduce extraction length by 1 "second"
        self.extract_length -= 1
        
        # Calculate reward
        
        
        drop_size_err = 1 - (self.state[0] - self.wanted_size)**2 # reward irgendwie anders
        
        rpm_cost = 1-((self.state[2]-300)*(1/300))
        
        solvent_cost = 1-((self.state[3]-25)*0.1)
        
        #if (action[0] == 0) or (int(round(action[1],0)) == 0):
        #    action_cost = 0
        #else:
        action_cost = 0#-1
        
        
        
        reward = reward + self.reward_weights[0]*(drop_size_err) + self.reward_weights[1]*rpm_cost + self.reward_weights[2]*solvent_cost + self.reward_weights[3]*action_cost
        #gewichtung testen
        
        #self.state['state'] == 0 means flooding
        if self.state[1] == 0:
            reward = -1
            #reward = reward - 100
        # Check if extraction is done
        # if drop is too small or to big: End the run
        if self.extract_length <= 0: 
            done = True
        elif (self.state[3] < 25) or (self.state[3] > 35):
            #reward = reward - 50 - self.extract_length
            reward = -1
            done = True
        elif (self.state[2] <= 300) or (self.state[2] >= 550):
            #reward = reward - 50 - self.extract_length
            reward = -1
            done = True
        
        
        # Apply state noise
        #self.state['drop_size'] += random.uniform(-0.05,0.05)
        
        # Set placeholder for info
        info = {}
         
        # Return step information
        return np.array(self.state).astype(np.float32), reward, done, info
    
    def reset(self):
        # Reset extraction state with some randomness
        #self.drop_size = 1.2
        rpm = random.randrange(300, 550, 10)
        solvent = round(random.uniform(25,35))

        self.state = [0,0,rpm,solvent]
        
        #print('solvent {}'.format(solvent))
        flooding = round(self.data['solvent {}'.format(solvent)]['flooding'][self.data['solvent {}'.format(solvent)]['rpm'].index(round(rpm/50)*50)])
        #self.drop_size = random.choice([2.5,1.5])
        Omega = self.omega['solvent {}'.format(solvent)]
        #print(Omega)
        drop_omega = 0
        for t in range(len(Omega)):
            drop_omega = drop_omega + Omega[t][0]*self.state[2]**t

        self.state = [round(drop_omega + random.uniform(-0.10, 0.10),3), flooding, rpm,solvent]
        # Reset extraction time, flooding time, out of DOS time
        self.extract_length = 200
        self.flooding_time = 0
        
        return np.array(self.state).astype(np.float32)
    
    def set_reset(self, start_rpm, start_solvent):
        
        
        
        rpm = start_rpm
        solvent = start_solvent
        self.state = [0,0,rpm,solvent]
        
        flooding = round(self.data['solvent {}'.format(solvent)]['flooding'][self.data['solvent {}'.format(solvent)]['rpm'].index(round(rpm/50)*50)])
        Omega = self.omega['solvent {}'.format(solvent)]
        drop_omega = 0
        for t in range(len(Omega)):
            drop_omega = drop_omega + Omega[t][0]*self.state[2]**t

        self.extract_length = 200
        self.flooding_time = 0
        
        self.state = [round(drop_omega + random.uniform(-0.10, 0.10),3), flooding, rpm,solvent]
        return np.array(self.state).astype(np.float32)

#%%
#environment with continuous action space
class ExtractionEnv_con(Env):
    #reward_weights = [drop_size_weight, rpm_weight, solvent_weight, action_weight]
    def __init__(self, wanted_size, omega, data, reward_weights):
        
        #parameter for reward calculation
        #self.flooding_time = 0
        #action space in diskret umbauen!
        self.action_space = gym.spaces.Box(low = np.array([-20,-2]), high = np.array([20,+2]), shape=(2,), dtype=np.float32)
                                     
        #[drop,flooding,rpm,solvent]
        self.observation_space = gym.spaces.Box(low = np.array([0,0,200,25]), high = np.array([2,1,600,35], dtype=np.float32))
        self.data = data
        self.omega = omega
        self.wanted_size = wanted_size
        self.reward_weights = reward_weights
        # Set start 
        # randomize the dropsize
        # calculation of the rpm in dependence of the drop size
       
        # Set Extraction Length
        self.extract_length = 200
    def step(self, action):
        # Apply action: Change rpm or solvent flow
        # 28.02.22: Change of the reward: should be between -1 and 1
        reward = 0
        done= False
        self.state[2] += action[0]
       
        self.state[3] += int(round(action[1],0)) 
            
        if (self.state[3] < 25) or (self.state[3] > 35):
            done = True
            reward = -1
            #reward = -50 - self.extract_length
            if self.state[3] < 25:
                Omega = self.omega['solvent 25']
                self.state[3] = 25
            else:
                Omega = self.omega['solvent 35']
                self.state[3] = 35
        else:
            Omega = self.omega['solvent {}'.format(self.state[3])]

        
        # update other parameters
        drop_omega = 0
        for t in range(len(Omega)):
            drop_omega = drop_omega + Omega[t]*self.state[2]**t
        
        self.state[0] = round(drop_omega[0] + random.uniform(-0.10, 0.10) ,3)
        
        self.state[1] = round(self.data['solvent {}'.format(self.state[3])]['flooding'][self.data['solvent {}'.format(self.state[3])]['rpm'].index(round(self.state[2]/50)*50)])
        
        
        # Reduce extraction length by 1 "second"
        self.extract_length -= 1
        
        # Calculate reward
        
        
        drop_size_err = 1 - (self.state[0] - self.wanted_size)**2 # reward irgendwie anders
        
        rpm_cost = 1-((self.state[2]-300)*(1/300))
        
        solvent_cost = 1-((self.state[3]-25)*0.1)
        
        #if (action[0] == 0) or (int(round(action[1],0)) == 0):
        #    action_cost = 0
        #else:
        action_cost = 0#-1
        
        
        
        reward = reward + self.reward_weights[0]*(drop_size_err) + self.reward_weights[1]*rpm_cost + self.reward_weights[2]*solvent_cost + self.reward_weights[3]*action_cost
        #gewichtung testen
        
        #self.state['state'] == 0 means flooding
        if self.state[1] == 0:
            reward = -1
            done = True
            #reward = reward - 100
        # Check if extraction is done
        # if drop is too small or to big: End the run
        if self.extract_length <= 0: 
            done = True
        elif (self.state[3] < 25) or (self.state[3] > 35):
            #reward = reward - 50 - self.extract_length
            reward = -1
            done = True
        elif (self.state[2] <= 300) or (self.state[2] >= 550):
            #reward = reward - 50 - self.extract_length
            reward = -1
            done = True
        
        
        # Apply state noise
        #self.state['drop_size'] += random.uniform(-0.05,0.05)
        
        # Set placeholder for info
        info = {}
         
        # Return step information
        return np.array(self.state).astype(np.float32), reward, done, info
    
    def reset(self):
        # Reset extraction state with some randomness
        #self.drop_size = 1.2
        rpm = random.randrange(300, 550, 10)
        solvent = round(random.uniform(25,35))
        self.state = [0,0,rpm,solvent]
        
        #print('solvent {}'.format(solvent))
        flooding = round(self.data['solvent {}'.format(solvent)]['flooding'][self.data['solvent {}'.format(solvent)]['rpm'].index(round(rpm/50)*50)])
        #self.drop_size = random.choice([2.5,1.5])
        Omega = self.omega['solvent {}'.format(solvent)]
        #print(Omega)
        drop_omega = 0
        for t in range(len(Omega)):
            drop_omega = drop_omega + Omega[t][0]*self.state[2]**t
       
        self.state = [round(drop_omega + random.uniform(-0.10, 0.10),3), flooding, rpm,solvent]
        # Reset extraction time, flooding time, out of DOS time
        self.extract_length = 200
        self.flooding_time = 0
        
        return np.array(self.state).astype(np.float32)
    
    def set_reset(self, start_rpm, start_solvent):
        
        rpm = start_rpm
        solvent = start_solvent
        self.state = [0,0,rpm,solvent]
        
        flooding = round(self.data['solvent {}'.format(solvent)]['flooding'][self.data['solvent {}'.format(solvent)]['rpm'].index(round(rpm/50)*50)])
        Omega = self.omega['solvent {}'.format(solvent)]
        drop_omega = 0
        for t in range(len(Omega)):
            drop_omega = drop_omega + Omega[t][0]*self.state[2]**t
       
        self.extract_length = 200
        self.flooding_time = 0
        
        self.state = [round(drop_omega + random.uniform(-0.10, 0.10),3), flooding, rpm,solvent]
        return np.array(self.state).astype(np.float32)
    
#%%
# import working models

DQN = tf.keras.models.load_model(r'C:/Users/marvi/sciebo/Masterarbeit/Python Module/DQN_model_28_02_22_1.h5')

PPO = PPO.load(r"C:/Users/marvi/ppo_1_03_03_22_1.zip")
#PPO = torch.load(r"C:/Users/marvi/PPO_test/policy.pth")
DDPG =  tf.keras.models.load_model(r'C:/Users/marvi/sciebo/Masterarbeit/Python Module/DDPG_model_14_02_22_4.h5')
#DDPG =  tf.keras.models.load_model(r'C:/Users/marvi/DDPG_model_02_03_22_1.h5')

with open(r'C:/Users/marvi/sciebo/Masterarbeit/Python Programme/Q_value_table_rew_60_20_20_100_gamma_90_epi_50000 26_01_22_2.pkl', 'rb') as f:
            Q_table = pickle.load(f)
            
#%%
# wanted_size = 1
# theta = SourceFileLoader("Theta.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/Theta.py").load_module()
# reward_weights = [0.6,0.2,0.2,1]
# with open(r'C:/Users/marvi/sciebo/Masterarbeit/Python Module/Data_3.pkl', 'rb') as f: #try could be useful
#     data = pickle.load(f)
    
# data_2, dummie = model_param.model_parameter(data['drop_size'], data['rpm'], data['solvent'], data['flooding'])

# all_theta = {}
# for key in data_2.keys():
#     #print(key)
#     all_theta[key]=theta.Theta(data_2,1,'rpm','drop_size', key)  

# dis_env = ExtractionEnv_dis(wanted_size, all_theta, data_2, reward_weights)
# con_env = ExtractionEnv_con(wanted_size, all_theta, data_2, reward_weights)
# teststate = dis_env.set_reset(310,25)
# teststate = np.reshape(teststate, [1, -1])
# new_s = copy.copy(teststate[0])
# new_s[0]=new_s[0]/2
# new_s[2]=new_s[2]/600
# new_s[3]=new_s[3]/35
# new_s = np.reshape(new_s, [1,-1])

# PPO.predict(new_s)
#%% 
episodes = 1
wanted_size = 1
theta = SourceFileLoader("Theta.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/Theta.py").load_module()
reward_weights = [0.6,0.2,0.2,1]
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Python Module/Data_3.pkl', 'rb') as f: #try could be useful
    data = pickle.load(f)
model_param = SourceFileLoader("model_param.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/model_param.py").load_module()

data_2, dummie = model_param.model_parameter(data['drop_size'], data['rpm'], data['solvent'], data['flooding'])

all_theta = {}
for key in data_2.keys():
    #print(key)
    all_theta[key]=theta.Theta(data_2,1,'rpm','drop_size', key)  

dis_env = ExtractionEnv_dis(wanted_size, all_theta, data_2, reward_weights)
con_env = ExtractionEnv_con(wanted_size, all_theta, data_2, reward_weights)
#%%DQN
for episode in range(1, episodes+1):
    teststate = dis_env.set_reset(310,25)
    done = False
    score = 0
    DQNdrop = []
    DQNrpm  = []
    DQNflooding = []
    DQNsolvent = []
    while not done:
        teststate = np.reshape(teststate, [1, -1])
        print(teststate[0])
        
        DQNdrop.append(teststate[0][0])
        DQNrpm.append(teststate[0][2])
        DQNflooding.append(teststate[0][1])
        DQNsolvent.append(teststate[0][3])
        new_s = copy.copy(teststate[0])
        new_s[0]=new_s[0]/2
        new_s[2]=new_s[2]/600
        new_s[3]=new_s[3]/35
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
    teststate = con_env.set_reset(350,25)
    done = False
    score = 0
    DDPGdrop = []
    DDPGrpm  = []
    DDPGflooding = []
    DDPGsolvent = []
    while not done:
        teststate = np.reshape(teststate, [1, -1])
        print(teststate[0])
        
        DDPGdrop.append(teststate[0][0])
        DDPGrpm.append(teststate[0][2])
        DDPGflooding.append(teststate[0][1])
        DDPGsolvent.append(teststate[0][3])
        new_s = copy.copy(teststate[0])
        new_s[0]=new_s[0]/2
        new_s[2]=new_s[2]/600
        new_s[3]=new_s[3]/35
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
    teststate = con_env.set_reset(530,35)
    done = False
    score = 0
    PPOdrop = []
    PPOrpm  = []
    PPOflooding = []
    PPOsolvent = []
    while not done:
        teststate = np.reshape(teststate, [1, -1])
        print(teststate[0])
        
        PPOdrop.append(teststate[0][0])
        PPOrpm.append(teststate[0][2])
        PPOflooding.append(teststate[0][1])
        PPOsolvent.append(teststate[0][3])
        new_s = copy.copy(teststate[0])
        new_s[0]=new_s[0]/2
        new_s[2]=new_s[2]/600
        new_s[3]=new_s[3]/35
        new_s = np.reshape(new_s, [1,-1])
        action = PPO.predict(new_s)[0]
        action[0][0]*=20
        action[0][1]*=2
        del new_s
        print(action)
        teststate, reward, done, info = con_env.step(action[0])
        #print(state)
        score+=reward
        
#%%Q_table
for episode in range(1, episodes+1):
    teststate = dis_env.set_reset(310,25)
    done = False
    score = 0
    Q_tabledrop = []
    Q_tablerpm  = []
    Q_tableflooding = []
    Q_tablesolvent = []
    while not done:
        teststate = np.reshape(teststate, [1, -1])
        print(teststate[0])
        
        Q_tabledrop.append(teststate[0][0])
        Q_tablerpm.append(teststate[0][2])
        Q_tableflooding.append(teststate[0][1])
        Q_tablesolvent.append(teststate[0][3])
        
        action_df = Q_table.loc[(Q_table['drop_size']==round(float(teststate[0][0]),2))]
        n = 0
        #print(action_df)
        while len(action_df)==0:
            n+=1
            action_df = Q_table.loc[(Q_table['drop_size']<=(teststate[0][0]+0.1*n))&(Q_table['drop_size']>=(teststate[0][0]-0.1*n)) ]#macht nicht das was es soll
        action_df = action_df.iloc[((action_df['solvent']-teststate[0][3]).abs().argsort()[:5])]
        #print(action_df)
        
        #----verbesserung aus Test Q-value Table 24_01_22
        action_vektor = np.zeros(16)
        
        for item in zip(action_df['Q_value'],action_df['rpm']):
            #print('Q-value/rpm: {}'.format(item))
            if np.count_nonzero(item[0])>2:
                Q_value_weighted = item[0]*((teststate[0][2]-np.abs(item[1]-teststate[0][2]))/teststate[0][2])
                #print('Q-value weighted: {}'.format(Q_value_weighted))
                action_vektor = action_vektor + Q_value_weighted
        #print('Action Matrix: {}'.format(action_vektor))
        
        action = np.argmax(action_vektor)
        if max(action_vektor) == 0:
            print('Action is 5 due to an empty action vector!')
            action = 5
        
        print(action)
        teststate, reward, done, info = dis_env.step(action)
        #print(state)
        score+=reward
        
#%%diagramms
import matplotlib.pyplot as plt
time = np.arange(len(DDPGdrop))+1
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
ax1.set_ylabel('rpm [1/min]',size=16)
ax1.plot(time, DQNsolvent, color='tab:red')
ax1.plot(time, DDPGsolvent, color='tab:orange')
ax1.plot(time, PPOsolvent, color='tab:purple')
ax1.plot(time, Q_tablesolvent, color='tab:pink')
ax1.tick_params(axis='y')
ax1.axis([1, 200,20, 40 ])