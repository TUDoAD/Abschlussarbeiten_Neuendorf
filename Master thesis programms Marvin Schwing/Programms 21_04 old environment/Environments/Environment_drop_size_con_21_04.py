# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:20:10 2022

@author: marvi
"""
from gym import Env
import gym
#from importlib.machinery import SourceFileLoader
import random
import numpy as np

#model_param = SourceFileLoader("model_param.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/model_param.py").load_module()


class ExtractionEnv_con(Env):
    #reward_weights = [drop_size_weight, rpm_weight, solvent_weight, action_weight]
    def __init__(self, wanted_drop_size, omega, data, reward_weights):
        
        #parameter for reward calculation
        #self.flooding_time = 0
        #self.drop_size = random.uniform(1,4)
        self.action_space = gym.spaces.Box(low = np.array([-50,-2]), high = np.array([50,+2]), shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low = np.array([0,0,200,20]), high = np.array([2,1,700,40], dtype=np.float32))
        self.data = data
        self.omega = omega
        self.wanted_size = wanted_drop_size
        self.reward_weights = reward_weights
        # Set start
        # Set Extraction Length
        self.extract_length = 200
    def step(self, action):
        # Apply action: Change rpm or solvent flow
        reward = 0
        info = False
        done = False
        solvent_for_flooding = self.state[3]
        self.state[2] += action[0]
        self.state[3] += int(round(action[1],0)) 
            
        if (self.state[3] < 25) or (self.state[3] > 35):
            #done = True
            #reward = -1
            if self.state[3] < 25:
                solvent_for_flooding = 25
                Omega = self.omega['solvent 25']
            else:
                solvent_for_flooding = 35
                Omega = self.omega['solvent 35']
        else:
            Omega = self.omega['solvent {}'.format(self.state[3])]

        
        # update other parameters
        drop_omega = 0
        for t in range(len(Omega)):
            drop_omega = drop_omega + Omega[t]*self.state[2]**t
        
        self.state[0] = round(drop_omega + random.uniform(-0.10, 0.10) ,2)
        
        self.state[1] = round(self.data['solvent {}'.format(solvent_for_flooding)]['flooding'][self.data['solvent {}'.format(solvent_for_flooding)]['rpm'].index(round(self.state[2]/50)*50)])
        
        
        # Reduce extraction length by 1 "second"
        self.extract_length -= 1
        
        # Calculate reward
        
        
        drop_size_err = 1 - (self.state[0] - self.wanted_size)**2 # reward irgendwie anders
        
        rpm_cost = 1-((self.state[2]-300)*(1/300))
        if rpm_cost>1:
            rpm_cost = 1
        #solvent_cost = 1-self.state['solvent']/35
        solvent_cost = 1-((self.state[3]-25)*0.1)
        if solvent_cost>1:
            solvent_cost = 1
        
        action_cost = 0
        reward = reward + self.reward_weights[0]*(drop_size_err) + self.reward_weights[1]*rpm_cost + self.reward_weights[2]*solvent_cost + self.reward_weights[3]*action_cost
        #gewichtung testen
        
        #self.state['state'] == 0 means flooding
        if self.state[1] == 0:
            reward = -1
        # Check if extraction is done
        # if drop is too small or to big: End the run
        if self.extract_length <= 0: 
            done = True
            info = True
        elif (self.state[3] < 25) or (self.state[3] > 35):
            reward =-1
            done = True
        elif (self.state[2] <= 300) or (self.state[2] >= 550):
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
        #self.drop_size = 1.2
        rpm = random.randrange(350, 550, 50)
        solvent = round(random.uniform(25,35))
        self.state = [0,0,rpm,solvent]
        #print('solvent {}'.format(solvent))
        flooding = round(self.data['solvent {}'.format(solvent)]['flooding'][self.data['solvent {}'.format(solvent)]['rpm'].index(round(rpm/50)*50)])
        #self.drop_size = random.choice([2.5,1.5])
        Omega = self.omega['solvent {}'.format(solvent)]
        #print(Omega)
        drop_omega = 0
        for t in range(len(Omega)):
            drop_omega = drop_omega + Omega[t]*self.state[2]**t
        #print(drop_omega[0])
        self.state[0]=round(drop_omega + random.uniform(-0.10, 0.10),3)
        self.state[1]=flooding
        
        # Reset extraction time, flooding time, out of DOS time
        self.extract_length = 200
        #self.flooding_time = 0
        
        return np.array(self.state).astype(np.float32)
    
    def set_reset(self, start_rpm, start_solvent):
        rpm = start_rpm
        solvent = start_solvent
        self.state = [0,0,rpm,solvent]
        #print('solvent {}'.format(solvent))
        flooding = round(self.data['solvent {}'.format(solvent)]['flooding'][self.data['solvent {}'.format(solvent)]['rpm'].index(round(rpm/50)*50)])
        #self.drop_size = random.choice([2.5,1.5])
        Omega = self.omega['solvent {}'.format(solvent)]
        #print(Omega)
        drop_omega = 0
        for t in range(len(Omega)):
            drop_omega = drop_omega + Omega[t]*self.state[2]**t
        #print(drop_omega[0])
        self.state[0]=round(drop_omega + random.uniform(-0.10, 0.10),3)
        self.state[1]=flooding
        
        # Reset extraction time, flooding time, out of DOS time
        self.extract_length = 200
        #self.flooding_time = 0
        
        return np.array(self.state).astype(np.float32)    
    

