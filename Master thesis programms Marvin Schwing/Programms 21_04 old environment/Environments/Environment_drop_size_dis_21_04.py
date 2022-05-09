# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:20:10 2022

@author: marvi
"""
from gym import Env
from gym.spaces import Discrete
import gym
#from importlib.machinery import SourceFileLoader
import random
import numpy as np

#model_param = SourceFileLoader("model_param.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/model_param.py").load_module()


class ExtractionEnv_dis(Env):
    #reward_weights = [drop_size_weight, rpm_weight, solvent_weight, action_weight]
    def __init__(self, wanted_drop_size, omega, data, reward_weights, rpm_scale):
        
        #parameter for reward calculation
        #self.flooding_time = 0
        #self.drop_size = random.uniform(1,4)
        self.action_space = Discrete(16)
        self.observation_space = gym.spaces.Box(low = np.array([0,0,200,20]), high = np.array([2,1,700,40], dtype=np.float32))
        #self.observation_space = gym.spaces.Dict({'drop_size': Box(low=0, high=5, shape=(1,)),
        #        'state': Discrete(2), 'rpm':  Box(low=200, high=600, shape=(1,)), 
        #        'solvent': Discrete(60), 'feed': Discrete(60)
        #        })
        self.data = data
        self.omega = omega
        self.wanted_size = wanted_drop_size
        self.reward_weights = reward_weights
        self.rpm_scale = rpm_scale
        # Set start 
        # randomize the dropsize
        # calculation of the rpm in dependence of the drop size
        #self.state = dict({'drop_size': round(self.drop_size,2),
        #                   'state': 1, #only placeholder
        #                   'rpm': 400, #only placeholder
        #                   'solvent': 30
        #                  })
        # Set Extraction Length
        self.extract_length = 200
    def step(self, action):
        # Apply action: Change rpm or solvent flow
        reward = 0
        info = False
        done = False
        solvent_for_flooding = self.state[3]
        if action < 11:
            self.state[2] += (action-5)*self.rpm_scale #kleinere änderung beim nächsten mal!!!!!
        if action >= 11:                    #25.02 rpm schritte verkleinert
            self.state[3] += (action-13)
            
        if (self.state[3] < 25) or (self.state[3] > 35):
            #done = True
            #reward = -1
            if self.state[3] < 25:
                Omega = self.omega['solvent 25']
                solvent_for_flooding = 25
            else:
                Omega = self.omega['solvent 35']
                solvent_for_flooding = 35
        else:
            Omega = self.omega['solvent {}'.format(self.state[3])]

        
        # update other parameters
        drop_omega = 0
        for t in range(len(Omega)):
            drop_omega = drop_omega + Omega[t]*self.state[2]**t
        
        #self.state[0] = round(drop_omega + random.uniform(-0.10, 0.10) ,2)
        
        self.state[0]=round(drop_omega, 2)
        
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
        
        if (action == 5) or (action == 13):
            action_cost = 0
        else:
            action_cost = -1
        
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
        rpm = random.randrange(300, 550, 50)
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
        #self.state[0]=round(drop_omega + random.uniform(-0.10, 0.10),3)
        #uncomment to get rid of the randomness
        self.state[0]=round(drop_omega,2)
        self.state[1]=flooding
        #self.state = dict({'drop_size': round(drop_omega[0] + random.uniform(-0.10, 0.10),2),
        #                   'state': flooding, 
        #                   'rpm': rpm,
        #                   'solvent': solvent
        #                  })
        
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
        #self.state[0]=round(drop_omega + random.uniform(-0.10, 0.10),2)
        
        self.state[0]=round(drop_omega,2)
        self.state[1]=flooding
        #self.state = dict({'drop_size': round(drop_omega[0] + random.uniform(-0.10, 0.10),2),
        #                   'state': flooding, 
        #                   'rpm': rpm,
        #                   'solvent': solvent
        #                  })
        
        # Reset extraction time, flooding time, out of DOS time
        self.extract_length = 200
        #self.flooding_time = 0
        
        return np.array(self.state).astype(np.float32)    
    

#%%
# wanted_drop_size=1
# #set own reward_weights
# reward_weights = [1,0.3,0.3,0]
# #load data for environment creation
# with open(r'C:/Users/marvi/sciebo/Masterarbeit/Programms 21_04 old environment/Environments/Environment data/theta_old_env.pkl', 'rb') as f: 
#     theta = pickle.load(f)
# with open(r'C:/Users/marvi/sciebo/Masterarbeit/Programms 21_04 old environment/Environments/Environment data/excel_data_old_env.pkl', 'rb') as f: 
#     excel_data = pickle.load(f)
# env = ExtractionEnv_dis()

