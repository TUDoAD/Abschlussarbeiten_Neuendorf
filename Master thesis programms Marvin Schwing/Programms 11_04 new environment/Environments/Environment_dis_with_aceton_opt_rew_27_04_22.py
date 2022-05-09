# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:39:40 2022

@author: marvi
"""
from gym import Env
#from gym.spaces import Discrete, Box
#from gym import spaces
import gym
import numpy as np
#import pickle
#from importlib.machinery import SourceFileLoader
import random
#import tensorflow as tf
#import stable_baselines3
#import copy


class ExtractionEnv_dis(Env):
    #reward_weights = [drop_size_weight, rpm_weight, solvent_weight, action_weight]
    def __init__(self, wanted_acetone, omega, data, reward_weights, rpm_scale):
        
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
        self.rpm_scale = rpm_scale
        # Set start 
        # randomize the dropsize
        # calculation of the rpm in dependence of the drop size
        # Set Extraction Length
        self.extract_length = 200
    def step(self, action):
        # Apply action: Change rpm or solvent flow
        # 28.02.22: Change of the reward: should be between -1 and 1
        reward = 0
        info = False
        done = False
        if action < 11:
            self.state[2] += (action-5)*self.rpm_scale #kleinere änderung beim nächsten mal!!!!!
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
        acetone_cost = 1 - np.abs(self.state[4] - self.wanted_acetone)
        if self.state[4]<self.wanted_acetone: #änderung 18.03.22
            acetone_cost = 2 #änderung 22_4_22
        rpm_cost = 1-((self.state[2]-300)*(1/300))
        if rpm_cost>1:
            rpm_cost = 1
        solvent_cost = 1-((self.state[3]-10)*0.1)
        if solvent_cost>1:
            solvent_cost = 1
        if (action == 5) or (action == 13):
            action_cost = 0
        else:
            action_cost = -1
        
        reward = reward + self.reward_weights[0]*(acetone_cost) + self.reward_weights[1]*rpm_cost + self.reward_weights[2]*solvent_cost + self.reward_weights[3]*action_cost
        #gewichtung testen
        
        if self.state[1] == 0:
            reward = -1
            #reward = reward - 100
        # Check if extraction is done
        # if drop is too small or to big: End the run
        if self.extract_length <= 0: 
            done = True
            info = True
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
        #info = {}
         
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
        self.extract_length = 200
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
        
        self.extract_length = 200
        self.flooding_time = 0
        
        return np.array(self.state).astype(np.float32)