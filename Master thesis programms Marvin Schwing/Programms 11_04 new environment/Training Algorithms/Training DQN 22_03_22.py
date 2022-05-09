# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:20:10 2022

@author: marvi
"""
from gym import Env
from gym.spaces import Discrete, Box
from gym import spaces
import gym
import numpy as np
import pickle
from importlib.machinery import SourceFileLoader
import random


class ExtractionEnv_dis(Env):
    #reward_weights = [drop_size_weight, rpm_weight, solvent_weight, action_weight]
    def __init__(self, wanted_acetone, omega, data, reward_weights):
        
        #parameter for reward calculation
        #self.flooding_time = 0
        #action space in diskret umbauen!
        self.action_space = gym.spaces.Discrete(16)
        #self.action_space = spaces.Box(low = np.array([-20,-2]), high = np.array([20,+2]), shape=(2,), dtype=np.float32)                                        
                                                    #[drop,flooding,rpm,solvent]
        self.observation_space = gym.spaces.Box(low = np.array([0,0,200,5,0]), high = np.array([2,1,700,25,5], dtype=np.float32))
        self.data = data
        self.omega = omega
        self.wanted_acetone = wanted_acetone
        self.reward_weights = reward_weights
        self.solvent_list = list(omega.keys())
        # Set start 
        # randomize the dropsize
        # calculation of the rpm in dependence of the drop size
        # Set Extraction Length
        self.extract_length = 100
    def step(self, action):
        # Apply action: Change rpm or solvent flow
        # 28.02.22: Change of the reward: should be between -1 and 1
        reward = 0
        done= False
        if action < 11:
            self.state[2] += (action-5)*4 #kleinere änderung beim nächsten mal!!!!!
        if action >= 11:                    #25.02 rpm schritte verkleinert
            self.state[3] += (action-13)
        
        if ('solvent {}'.format(self.state[3]) in self.solvent_list) == True:
            Omega = self.omega['solvent {}'.format(self.state[3])]
        else:
            Omega_around = []
            solvent_for_int = []
            n = 1
            while len(Omega_around)<2:
                
                if ('solvent {}'.format(self.state[3]+n) in self.solvent_list) == True:
                    Omega_around.append(self.omega['solvent {}'.format(self.state[3]+n)])
                    solvent_for_int.append(self.state[3]+n)
                if ('solvent {}'.format(self.state[3]-n) in self.solvent_list) == True:
                    Omega_around.append(self.omega['solvent {}'.format(self.state[3]-n)])
                    solvent_for_int.append(self.state[3]-n)
                n +=1
            
            #lineare interpolation der omega parameter
            Omega = []
            Omega.append(Omega_around[0][0]+(Omega_around[1][0]-Omega_around[0][0])/(solvent_for_int[1]-solvent_for_int[0])*(self.state[3]-solvent_for_int[0]))
            Omega.append(Omega_around[0][1]+(Omega_around[1][1]-Omega_around[0][1])/(solvent_for_int[1]-solvent_for_int[0])*(self.state[3]-solvent_for_int[0]))
        # update other parameters
        drop_omega = 0
        
        for t in range(len(Omega)):
            drop_omega = drop_omega + Omega[t]*self.state[2]**t
        
        self.state[0] = round(drop_omega[0] + random.uniform(-0.10, 0.10) ,3)
        
        #lineare interpolation der aceton konzentration
        if ('solvent {}'.format(self.state[3]) in self.solvent_list) == True:
            #search for the index
            for item in self.data['solvent {}'.format(self.state[3])]['rpm']:
                if item > self.state[2]:
                    if item == 300:
                        x_1 = 0 
                        x_2 = 1
                        self.state[1] = self.data['solvent {}'.format(self.state[3])]['flooding'][x_1]
                        break
                    else:
                        x_2 = self.data['solvent {}'.format(self.state[3])]['rpm'].index(item)
                        x_1 = x_2-1
                        if abs(self.data['solvent {}'.format(self.state[3])]['rpm'][x_2]-self.state[2])<abs(self.data['solvent {}'.format(self.state[3])]['rpm'][x_1]-self.state[2]):
                            self.state[1] = self.data['solvent {}'.format(self.state[3])]['flooding'][x_2]
                        else:
                            self.state[1] = self.data['solvent {}'.format(self.state[3])]['flooding'][x_1]
                        break
            if self.state[2]>=600:
                #find biggest index 
                x_2 = self.data['solvent {}'.format(self.state[3])]['rpm'].index(max(self.data['solvent {}'.format(self.state[3])]['rpm']))
                x_1 = x_2 -1
                self.state[1] = self.data['solvent {}'.format(self.state[3])]['flooding'][x_2]
            
            #rpm of the index for linear interpolation
            x_1_rpm = self.data['solvent {}'.format(self.state[3])]['rpm'][x_1]
            x_2_rpm = self.data['solvent {}'.format(self.state[3])]['rpm'][x_2]
                
            y_1 = self.data['solvent {}'.format(self.state[3])]['aceton_conc'][int(x_1)]
            y_2 = self.data['solvent {}'.format(self.state[3])]['aceton_conc'][int(x_2)]
            
            aceton_conc = y_1 + (y_2-y_1)/((x_2_rpm)-(x_1_rpm))*(self.state[2]-(x_1_rpm))
            
        else:
            
            solvent_around = []
            n = 1
            while len(solvent_around)<1:
                if self.state[3]<10:
                    solvent_around.append(12)
                if self.state[3]>20:
                    solvent_around.append(20)
                if ('solvent {}'.format(self.state[3]+n) in self.solvent_list) == True:
                    solvent_around.append(self.state[3]+n)
                n += 1
            solvent_around.append(solvent_around[0]-2)
            conc_aceton = []
            flooding_around = []
            for solvent in solvent_around:
                for item in self.data['solvent {}'.format(solvent)]['rpm']:
                    if item > self.state[2]:
                        if item == 300:
                            x_1 = 0 
                            x_2 = 1
                            flooding_around.append(self.data['solvent {}'.format(solvent)]['flooding'][x_1])
                            break
                        else:
                            x_2 = self.data['solvent {}'.format(solvent)]['rpm'].index(item)
                            x_1 = x_2-1
                            if abs(self.data['solvent {}'.format(solvent)]['rpm'][x_2]-self.state[2])<abs(self.data['solvent {}'.format(solvent)]['rpm'][x_1]-self.state[2]):
                                flooding_around.append(self.data['solvent {}'.format(solvent)]['flooding'][x_2])
                            else:
                                flooding_around.append(self.data['solvent {}'.format(solvent)]['flooding'][x_1])
                            break
                        
                if self.state[2]>=600:
                    #find biggest index 
                    x_2 = self.data['solvent {}'.format(solvent)]['rpm'].index(max(self.data['solvent {}'.format(solvent)]['rpm']))
                    x_1 = x_2 -1
                    flooding_around.append(self.data['solvent {}'.format(solvent)]['flooding'][x_2])
                #rpm of the index for linear interpolation
                x_1_rpm = self.data['solvent {}'.format(solvent)]['rpm'][x_1]
                x_2_rpm = self.data['solvent {}'.format(solvent)]['rpm'][x_2]
                y_1 = self.data['solvent {}'.format(solvent)]['aceton_conc'][int(x_1)]
                y_2 = self.data['solvent {}'.format(solvent)]['aceton_conc'][int(x_2)]
                
                conc_aceton.append(y_1 + (y_2-y_1)/((x_2_rpm)-(x_1_rpm))*(self.state[2]-(x_1_rpm)))
            
            aceton_conc = (conc_aceton[0]+conc_aceton[1])/2
            self.state[1] = np.round((flooding_around[0]+flooding_around[1])/2,0)
            
        self.state[4] = aceton_conc
        
        # Reduce extraction length by 1 "second"
        self.extract_length -= 1
        
        # Calculate reward
        acetone_cost = 1 - (self.state[4] - self.wanted_acetone)**2 # reward irgendwie anders
        if self.state[4]<self.wanted_acetone: #änderung 18.03.22
            acetone_cost = 2 #änderung 22_3_22
        rpm_cost = 1-((self.state[2]-300)*(1/300))
        if rpm_cost>1:
            rpm_cost = 1
        solvent_cost = 1-((self.state[3]-10)*0.1)
        if solvent_cost>1:
            solvent_cost = 1
        action_cost = 0#-1
        
        reward = reward + self.reward_weights[0]*(acetone_cost) + self.reward_weights[1]*rpm_cost + self.reward_weights[2]*solvent_cost + self.reward_weights[3]*action_cost
        #gewichtung testen
        
        if self.state[1] == 0:
            reward = -1
            #reward = reward - 100
        # Check if extraction is done
        # if drop is too small or to big: End the run
        if self.extract_length <= 0: 
            done = True
        elif (self.state[3] < 10) or (self.state[3] > 20):
            #reward = reward - 50 - self.extract_length
            reward = -1
            done = True
        elif (self.state[2] <= 250) or (self.state[2] >= 650):
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
        rpm = random.randrange(350, 550, 50)
        solvent = round(random.uniform(10,20))

        self.state = [0,0,rpm,solvent,0]
        
        if ('solvent {}'.format(self.state[3]) in self.solvent_list) == True:
            Omega = self.omega['solvent {}'.format(self.state[3])]
        else:
            Omega_around = []
            solvent_for_int = []
            n = 1
            while len(Omega_around)<2:
                if ('solvent {}'.format(self.state[3]+n) in self.solvent_list) == True:
                    Omega_around.append(self.omega['solvent {}'.format(self.state[3]+n)])
                    solvent_for_int.append(self.state[3]+n)
                if ('solvent {}'.format(self.state[3]-n) in self.solvent_list) == True:
                    Omega_around.append(self.omega['solvent {}'.format(self.state[3]-n)])
                    solvent_for_int.append(self.state[3]-n)
                
                #raise ValueError('A very specific bad thing happened.')
                n +=1
            #lineare interpolation der omega parameter
            Omega = []
            Omega.append(Omega_around[0][0]+(Omega_around[1][0]-Omega_around[0][0])/(solvent_for_int[1]-solvent_for_int[0])*(self.state[3]-solvent_for_int[0]))
            Omega.append(Omega_around[0][1]+(Omega_around[1][1]-Omega_around[0][1])/(solvent_for_int[1]-solvent_for_int[0])*(self.state[3]-solvent_for_int[0]))
        # update other parameters
        drop_omega = 0
        for t in range(len(Omega)):
            drop_omega = drop_omega + Omega[t]*self.state[2]**t
        
        self.state[0] = round(drop_omega[0] + random.uniform(-0.10, 0.10) ,3)
        
        if ('solvent {}'.format(self.state[3]) in self.solvent_list) == True:
            #search for the index
            for item in self.data['solvent {}'.format(self.state[3])]['rpm']:
                if item > self.state[2]:
                    if item == 300:
                        x_1 = 0 
                        x_2 = 1
                        self.state[1] = self.data['solvent {}'.format(self.state[3])]['flooding'][x_1]
                        break
                    else:
                        x_2 = self.data['solvent {}'.format(solvent)]['rpm'].index(item)
                        x_1 = x_2-1
                        if abs(self.data['solvent {}'.format(self.state[3])]['rpm'][x_2]-self.state[2])<abs(self.data['solvent {}'.format(self.state[3])]['rpm'][x_1]-self.state[2]):
                            self.state[1] = self.data['solvent {}'.format(self.state[3])]['flooding'][x_2]
                        else:
                            self.state[1] = self.data['solvent {}'.format(self.state[3])]['flooding'][x_1]
                        break
            if self.state[2]>=600:
                #find biggest index 
                x_2 = self.data['solvent {}'.format(solvent)]['rpm'].index(max(self.data['solvent {}'.format(solvent)]['rpm']))
                x_1 = x_2 -1
                self.state[1] = self.data['solvent {}'.format(self.state[3])]['flooding'][x_2]

            #rpm of the index for linear interpolation
            x_1_rpm = self.data['solvent {}'.format(self.state[3])]['rpm'][x_1]
            x_2_rpm = self.data['solvent {}'.format(self.state[3])]['rpm'][x_2]
                
            y_1 = self.data['solvent {}'.format(self.state[3])]['aceton_conc'][int(x_1)]
            y_2 = self.data['solvent {}'.format(self.state[3])]['aceton_conc'][int(x_2)]
            
            aceton_conc = y_1 + (y_2-y_1)/((x_2_rpm)-(x_1_rpm))*(self.state[2]-(x_1_rpm))
            
        else:
            
            solvent_around = []
            n = 1
            while len(solvent_around)<1:
                if self.state[3]<10:
                    solvent_around.append(12)
                if self.state[3]>20:
                    solvent_around.append(20)
                if ('solvent {}'.format(self.state[3]+n) in self.solvent_list) == True:
                    solvent_around.append(self.state[3]+n)
                n += 1
            solvent_around.append(solvent_around[0]-2)
            conc_aceton = []
            flooding_around = []
            for solvent in solvent_around:
                for item in self.data['solvent {}'.format(solvent)]['rpm']:
                    if item > self.state[2]:
                        if item == 300:
                            x_1 = 0 
                            x_2 = 1
                            flooding_around.append(self.data['solvent {}'.format(solvent)]['flooding'][x_1])
                            break
                        else:
                            x_2 = self.data['solvent {}'.format(solvent)]['rpm'].index(item)
                            x_1 = x_2-1
                            if abs(self.data['solvent {}'.format(solvent)]['rpm'][x_2]-self.state[2])<abs(self.data['solvent {}'.format(solvent)]['rpm'][x_1]-self.state[2]):
                                flooding_around.append(self.data['solvent {}'.format(solvent)]['flooding'][x_2])
                            else:
                                flooding_around.append(self.data['solvent {}'.format(solvent)]['flooding'][x_1])
                            break
                if self.state[2]>=600:
                    #find biggest index 
                    x_2 = self.data['solvent {}'.format(solvent)]['rpm'].index(max(self.data['solvent {}'.format(solvent)]['rpm']))
                    x_1 = x_2 -1
                    flooding_around.append(self.data['solvent {}'.format(solvent)]['flooding'][x_2])
    
                #rpm of the index for linear interpolation
                x_1_rpm = self.data['solvent {}'.format(solvent)]['rpm'][x_1]
                x_2_rpm = self.data['solvent {}'.format(solvent)]['rpm'][x_2]
                y_1 = self.data['solvent {}'.format(solvent)]['aceton_conc'][int(x_1)]
                y_2 = self.data['solvent {}'.format(solvent)]['aceton_conc'][int(x_2)]
                
                conc_aceton.append(y_1 + (y_2-y_1)/((x_2_rpm)-(x_1_rpm))*(self.state[2]-(x_1_rpm)))
            self.state[1] = np.round((flooding_around[0]+flooding_around[1])/2,0)
            aceton_conc = (conc_aceton[0]+conc_aceton[1])/2
        self.state[4] = aceton_conc
        
        # Reset extraction time, flooding time, out of DOS time
        self.extract_length = 100
        self.flooding_time = 0
        
        return np.array(self.state).astype(np.float32)
    
    def set_reset(self, start_rpm, start_solvent):
        
        
        
        rpm = start_rpm
        solvent = start_solvent
        self.state = [0,0,rpm,solvent,0]

        if ('solvent {}'.format(self.state[3]) in self.solvent_list) == True:
            Omega = self.omega['solvent {}'.format(self.state[3])]
        else:
            Omega_around = []
            solvent_for_int = []
            n = 1
            while len(Omega_around)<2:
                
                if ('solvent {}'.format(self.state[3]+n) in self.solvent_list) == True:
                    Omega_around.append(self.omega['solvent {}'.format(self.state[3]+n)])
                    solvent_for_int.append(self.state[3]+n)
                if ('solvent {}'.format(self.state[3]-n) in self.solvent_list) == True:
                    Omega_around.append(self.omega['solvent {}'.format(self.state[3]-n)])
                    solvent_for_int.append(self.state[3]-n)
                n +=1
            #lineare interpolation der omega parameter
            Omega = []
            Omega.append(Omega_around[0][0]+(Omega_around[1][0]-Omega_around[0][0])/(solvent_for_int[1]-solvent_for_int[0])*(self.state[3]-solvent_for_int[0]))
            Omega.append(Omega_around[0][1]+(Omega_around[1][1]-Omega_around[0][1])/(solvent_for_int[1]-solvent_for_int[0])*(self.state[3]-solvent_for_int[0]))
        # update other parameters
        drop_omega = 0
        for t in range(len(Omega)):
            drop_omega = drop_omega + Omega[t]*self.state[2]**t
        
        self.state[0] = round(drop_omega[0] + random.uniform(-0.10, 0.10) ,3)
        
        if ('solvent {}'.format(self.state[3]) in self.solvent_list) == True:
            #search for the index
            for item in self.data['solvent {}'.format(self.state[3])]['rpm']:
                if item > self.state[2]:
                    if item == 300:
                        x_1 = 0 
                        x_2 = 1
                        self.state[1] = self.data['solvent {}'.format(self.state[3])]['flooding'][x_1]
                        break
                    else:
                        x_2 = self.data['solvent {}'.format(solvent)]['rpm'].index(item)
                        x_1 = x_2-1
                        if abs(self.data['solvent {}'.format(self.state[3])]['rpm'][x_2]-self.state[2])<abs(self.data['solvent {}'.format(self.state[3])]['rpm'][x_1]-self.state[2]):
                            self.state[1] = self.data['solvent {}'.format(self.state[3])]['flooding'][x_2]
                        else:
                            self.state[1] = self.data['solvent {}'.format(self.state[3])]['flooding'][x_1]
                        break
            if self.state[2]>=600:
                #find biggest index 
                x_2 = self.data['solvent {}'.format(solvent)]['rpm'].index(max(self.data['solvent {}'.format(solvent)]['rpm']))
                x_1 = x_2 -1
                self.state[1] = self.data['solvent {}'.format(self.state[3])]['flooding'][x_2]

            #rpm of the index for linear interpolation
            x_1_rpm = self.data['solvent {}'.format(self.state[3])]['rpm'][x_1]
            x_2_rpm = self.data['solvent {}'.format(self.state[3])]['rpm'][x_2]
                
            y_1 = self.data['solvent {}'.format(self.state[3])]['aceton_conc'][int(x_1)]
            y_2 = self.data['solvent {}'.format(self.state[3])]['aceton_conc'][int(x_2)]
            
            
            aceton_conc = y_1 + (y_2-y_1)/((x_2_rpm)-(x_1_rpm))*(self.state[2]-(x_1_rpm))
            
        else:
            
            solvent_around = []
            n = 1
            while len(solvent_around)<1:
                if self.state[3]<10:
                    solvent_around.append(12)
                if self.state[3]>20:
                    solvent_around.append(20)
                if ('solvent {}'.format(self.state[3]+n) in self.solvent_list) == True:
                    solvent_around.append(self.state[3]+n)
                n += 1
            solvent_around.append(solvent_around[0]-2)
            conc_aceton = []
            flooding_around = []
            for solvent in solvent_around:
                for item in self.data['solvent {}'.format(solvent)]['rpm']:
                    if item > self.state[2]:
                        if item == 300:
                            x_1 = 0 
                            x_2 = 1
                            flooding_around.append(self.data['solvent {}'.format(solvent)]['flooding'][x_1])
                            break
                        else:
                            x_2 = self.data['solvent {}'.format(solvent)]['rpm'].index(item)
                            x_1 = x_2-1
                            if abs(self.data['solvent {}'.format(solvent)]['rpm'][x_2]-self.state[2])<abs(self.data['solvent {}'.format(solvent)]['rpm'][x_1]-self.state[2]):
                                flooding_around.append(self.data['solvent {}'.format(solvent)]['flooding'][x_2])
                            else:
                                flooding_around.append(self.data['solvent {}'.format(solvent)]['flooding'][x_1])
                            break
                if self.state[2]>=600:
                    #find biggest index 
                    x_2 = self.data['solvent {}'.format(solvent)]['rpm'].index(max(self.data['solvent {}'.format(solvent)]['rpm']))
                    x_1 = x_2 -1
                    flooding_around.append(self.data['solvent {}'.format(solvent)]['flooding'][x_2])
    
                #rpm of the index for linear interpolation
                x_1_rpm = self.data['solvent {}'.format(solvent)]['rpm'][x_1]
                x_2_rpm = self.data['solvent {}'.format(solvent)]['rpm'][x_2]
                y_1 = self.data['solvent {}'.format(solvent)]['aceton_conc'][int(x_1)]
                y_2 = self.data['solvent {}'.format(solvent)]['aceton_conc'][int(x_2)]
                
                
                conc_aceton.append(y_1 + (y_2-y_1)/((x_2_rpm)-(x_1_rpm))*(self.state[2]-(x_1_rpm)))
            self.state[1] = np.round((flooding_around[0]+flooding_around[1])/2,0) 
            aceton_conc = (conc_aceton[0]+conc_aceton[1])/2
        self.state[4] = aceton_conc
        
        self.extract_length = 100
        self.flooding_time = 0
        
        return np.array(self.state).astype(np.float32)
#%%
model_param = SourceFileLoader("new_model_param.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/new_model_param.py").load_module()
theta = SourceFileLoader("Theta.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/Theta.py").load_module()
#reward_weights = [0.9,0.05,0.05,1]
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Python Module/Data_new_aceton_18_03.pkl', 'rb') as f: 
    data = pickle.load(f)
    
data_2, dummie = model_param.model_parameter(data['drop_size'], data['rpm'], data['solvent'], data['flooding'], data['aceton_conc'])
all_theta = {}
for key in data_2.keys():
    #print(key)
    all_theta[key]=theta.Theta(data_2,1,'rpm','drop_size', key)

#%%
import tensorflow as tf
import copy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from collections import namedtuple
from collections import deque
import matplotlib.pyplot as plt
import datetime
#%%
class DQNAgent:
    # epsilon_greedy, epsilon_min, epsilon_decay decide if the action is random or model based
    # memory saves the namedtuple "Transition", max_memory_size gives the size of the memory
    def __init__(
            self, env, discount_factor = 0.99, epsilon_greedy = 1.0, epsilon_min=0.01, 
            epsilon_decay=0.9999, learning_rate=1e-3,max_memory_size = 100000):
        #self.enf = env
        self.state_size = 1
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=max_memory_size)
        self.gamma = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate
        self.num_states = 5
        self.num_actions = 16
        self.counter = 0
        self.X_shape = (self.num_states)
        self.hidden_sizes_1 = (500,200,100)
        self.model = self.ANN2(layer_sizes=list(self.hidden_sizes_1)+[self.num_actions])
    # use of the same model as DDPG for comparrison reasons
    def ANN2(self,layer_sizes, hidden_activation='relu'):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=self.X_shape))    
        for h in layer_sizes[:-1]:
            model.add(tf.keras.layers.Dense(units=h, activation='relu'))
        model.add(tf.keras.layers.Dense(units=16)) # 16 possible discrete actions between -10 to +10 rpm and -2 and +2 solvent
        model.compile(loss = 'mse', optimizer = Adam(lr = self.lr))
        return model
    
    def remember(self, transition, action):
        #print(len(self.memory))
        if len(self.memory)==100000:
            for i in range(len(self.memory)):
                if self.memory[len(self.memory)-1-i][1] == action:
                    #print(action)
                    #print(self.memory[len(self.memory)-1-i])
                    del self.memory[len(self.memory)-1-i]
                    break
        #print('Appended: {}'.format(transition))
        self.memory.append(transition)
        
    def choose_action(self, state):
        new_s = copy.copy(state[0])
        new_s[0]=new_s[0]/2
        new_s[2]=new_s[2]/600
        new_s[3]=new_s[3]/20
        new_s[4]=new_s[4]/2
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        new_s = np.reshape(new_s, [1,-1])
        q_values = self.model.predict(new_s)[0]
        return np.argmax(q_values)
    
    def _learn(self, batch_samples):   # rewards are clipped to -1-1 for stabilization reasons: Henderson et al. 2018 An Introduction to deep reinforcement learning
        batch_states, batch_targets = [], []
        for transition in batch_samples:
            s, a, r, next_s, done = transition
            if done:
                target = r
            else:
                #print(next_s)
                #self.model.summary()
                next_s = np.reshape(next_s,[1, -1])
                target = (r + self.gamma * np.amax(self.model.predict(next_s)[0]))
                #print('target')
                #print(target)
                #print('model.predict')
                #print(self.model.predict(next_s)[0])
                #print( np.amax(self.model.predict(next_s)[0]))
                
                #print('Saved next State: {}'.format(next_s))
                #print('Saved current State: {}'.format(s))
                #print('calculated Target: {}'.format(target))
                #print('Saved Reward: {}'.format(r))
                #print('prediction with current model (next State): {}'.format(self.model.predict(next_s)[0]))
                #print('max value: {}'.format(np.amax(self.model.predict(next_s)[0])))
                #print('prediction with current model (State): {}'.format(self.model.predict(s)[0]))
            s = np.reshape(s,[1, -1])
            target_all = self.model.predict(s)[0]
            #print('target_all')
            #print(target_all)
            target_all[a] = target
            #print(a)
            #print(target_all)
            #('s.flatten: {}'.format(s.flatten))
            batch_states.append(s.flatten())
            batch_targets.append(target_all)
            self._adjust_epsilon()
        self.model.fit(x=np.array(batch_states),y=np.array(batch_targets),epochs = 1, verbose = 0)
        #return self.model
    
    def _adjust_epsilon(self):
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.counter += 1
        if self.counter%100 == 0:
            print('current epsilon: {}'.format(self.epsilon))
            
    def replay(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        #print(samples)
        #print('above are samples')
        #print('these are the samples: {}'.format(samples))
        self._learn(samples)
        #return history.history['loss'][0]
    
    def get_model(self):
        return self.model
    
    def save_model(self):
        #self.model.save_weights('DQN_weights_04_03_22_1',overwrite = True)
        self.model.save(r'C:/Users/marvi/sciebo/Masterarbeit/Model/DQN'+ datetime.datetime.today().strftime('%Y_%m_%d %H_%M')+'.h5')
        
#%%
#EPISODES = 5 
batch_size = 64
init_replay_memory_size = 100000
early_stopping = deque(maxlen=5)
stop = False
Transition = namedtuple('Transition', ('state', 'action', 'reward','next_state', 'done'))
wanted_size = 1.2
if __name__ == '__main__':
    env = ExtractionEnv_dis(wanted_size, all_theta, data_2, [1,0.3,0.3,1]) 
    testEnv = ExtractionEnv_dis(wanted_size, all_theta, data_2, [1,0.3,0.3,1])
    agent = DQNAgent(env)
    #drop_avg = 0
    state = env.reset()
    #drop = state_curr[0]
    #for p in range(3):
    #    drop_avg = drop_avg + drop
    #    state_curr = env.step(5)
    #    drop = state_curr[0][0]
    #state = state_curr[0][:]
    #state[0] = round(drop_avg/3,3)
    #raise SystemExit("Stop right there!") 
    state = np.reshape(state, [1, -1])
    action_in_memory = np.zeros(16)
    ## initiliaze replay buffer
    for l in range(init_replay_memory_size):
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        
        #drop_avg = 0
        #reward_avg = 0
        #drop = next_state[0]
      
        #for p in range(3):
        #    drop_avg = drop_avg + drop
        #    reward_avg = reward_avg + reward
        #    next_state, reward, done, _ = env.step(5)
        #    drop = next_state[0]
            
        #next_state[0] = round(drop_avg/3,3)
        #reward = reward_avg/3
       
        #raise SystemExit("Stop right there!")

        next_state = np.reshape(next_state,[1, -1])
        new_s = copy.copy(state[0])
        new_s[0]=new_s[0]/2
        new_s[2]=new_s[2]/600
        new_s[3]=new_s[3]/20
        new_s[4]=new_s[4]/2
        new_s2 = copy.copy(next_state[0])
        new_s2[0]=new_s2[0]/2
        new_s2[2]=new_s2[2]/600
        new_s2[3]=new_s2[3]/20
        new_s2[4]=new_s2[4]/2
        if reward == -1:
            d = False
        else:
            d = True
        agent.remember(Transition(new_s, action, reward, new_s2, d),action)
        action_in_memory[action] += 1 
        #print(Transition(new_s, action, reward, new_s2, done))
        del new_s
        del new_s2
        if done:
            state = env.reset()
            state = np.reshape(state, [1, -1])
        else:
            state = next_state
        
    
    total_rewards, losses = [], []
    # for early stopping
    e = 1
    print('start learning')
    while stop == False:
        print(stop)
        #state = env.reset()
        
        #drop_avg = 0
        state_curr = env.reset()
        print(state_curr)
        #drop = state_curr[0]
        #for p in range(3):
        #    drop_avg = drop_avg + drop
        #    state_curr = env.step(5)
        #    drop = state_curr[0][0]
        #state = state_curr[0][:]
        #state[0] = round(drop_avg/3,3)
        #print(state)
        state = np.reshape(state, [1, -1])
        sum_reward = 0
        done = False
        i = 1
        while not done:
            action = agent.choose_action(state)
            #action_in_memory[action] += 1 
            
            next_state, reward, done, _ = env.step(action)
            
            #drop_avg = 0
            #reward_avg = 0
            #drop = next_state[0]
            #print(drop)
            #print(reward)
            #for p in range(3):
            #    drop_avg = drop_avg + drop
            #    reward_avg = reward_avg + reward
            #    next_state, reward, done, _ = env.step(5)
            #    drop = next_state[0]
                #print(reward)
                #print(drop)
            #next_state[0] = round(drop_avg/3,3)
            #reward = reward_avg/3
            print('Current state: {}'.format(next_state))
            print('Reward: {}'.format(reward))
            #print(next_state)
            #raise SystemExit("Stop right there!")
            
            next_state = np.reshape(next_state,[1, -1])
            new_s = copy.copy(state[0])
            new_s[0]=new_s[0]/2
            new_s[2]=new_s[2]/600
            new_s[3]=new_s[3]/20
            new_s[4]=new_s[4]/2
            new_s2 = copy.copy(next_state[0])
            new_s2[0]=new_s2[0]/2
            new_s2[2]=new_s2[2]/600
            new_s2[3]=new_s2[3]/20
            new_s2[4]=new_s2[4]/2
            
            if reward == -1:
                d = True
            else:
                d = False
            agent.remember(Transition(new_s, action, reward, new_s2, d),action)
            
            del new_s
            del new_s2
            sum_reward += reward
            state = next_state
            #print(i)
            if i % 10 == 0:
                print('Step: {}, Reward: {}/{}'.format(i,sum_reward,5+i*5))
                
                
            if done:
                test_model = agent.get_model()
                episodes = 1
                for episode in range(1, episodes+1):
                    teststate = testEnv.reset()
                    done = False
                    score = 0
                    steps = 0
                    while not done:
                        teststate = np.reshape(teststate, [1, -1])
                        new_s = copy.copy(teststate[0])
                        new_s[0]=new_s[0]/2
                        new_s[2]=new_s[2]/600
                        new_s[3]=new_s[3]/20
                        new_s[4]=new_s[4]/2
                        new_s = np.reshape(new_s, [1,-1])
                        print('new_s for prdiction: {}'.format(new_s))
                        action = test_model.predict(new_s)[0]
                        steps += 1
                        print(teststate)
                        print(np.argmax(action))
                        del new_s
                        #print(action)
                        action = np.argmax(action)
                        #print(action)
                        teststate, reward, done, info = testEnv.step(action)
                        #print(state)
                        score+=reward
                    if score > 100 and i > 150:
                        early_stopping.append(score)
                        first = early_stopping[0]
                        if len(early_stopping)==5:
                            for stopping in early_stopping:
                                if np.abs(first-stopping)<=10:
                                    stop = True
                                else:
                                    stop = False
                            if stop == True:
                                print('Early stopping criteria was met')
                                agent.save_model()
                    
                print('Actual Model peformance: {} with {} actions'.format(score,steps))
                #print(score)
                total_rewards.append(sum_reward)
                print('Episode ended after {} Steps'.format(i))
                print('Episode: %d, Total Reward: %d'
                         % (e, sum_reward))
                e += 1
            #loss = agent.replay(batch_size)
            #losses.append(loss)
            #if i % 10 == 0:
            print('Update Neuralnet')
            agent.replay(batch_size)
            i += 1
            #print('Actions in Memory: {}'.format(action_in_memory))
            #raise SystemExit("Stop right there!")
    agent.save_model()  
#%%
agent.save_model()


def plot_learning_history(history):
    fig = plt.figure(1, figsize=(14, 5))
    ax = fig.add_subplot(1, 1, 1)
    episodes = np.arange(len(history))+1
    plt.plot(episodes, history, lw=4,
             marker='o', markersize=10)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Episodes', size=20)
    plt.ylabel('Sum Rewards', size=20)
    plt.show()
    
plot_learning_history(total_rewards)
#%%
test_model = agent.get_model()
#%%
test_model.summary()
teststate = testEnv.reset()
new_s = copy.copy(teststate)
print(teststate)
new_s[0]=new_s[0]/2
new_s[2]=new_s[2]/600
new_s[3]=new_s[3]/20
new_s[4]=new_s[4]/2
new_s = np.reshape(new_s, [1,-1])
print(teststate)
print(new_s)
action = test_model.predict(new_s)
print(action)
#%%
#test_model.save_weights('DQN_weights_25_02_22_1',overwrite = True)
test_model.save(r'C:/Users/marvi/sciebo/Masterarbeit/Model/DQN'+ datetime.datetime.today().strftime('%Y_%m_%d %H_%M')+'.h5')