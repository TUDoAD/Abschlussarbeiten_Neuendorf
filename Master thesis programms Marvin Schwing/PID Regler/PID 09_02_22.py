# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 10:19:56 2022

@author: marvi
"""
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
import os
import sys

con_env = SourceFileLoader("Environment_drop_size_con_21_04_22.py", r"C:/Users/marvi/sciebo/Masterarbeit/Programms 21_04 old environment/Environments/Environment_drop_size_con_21_04.py").load_module()
wanted_drop_size=1
#set own reward_weights
reward_weights = [1,0.3,0.3,0]

#load data for environment creation
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Programms 21_04 old environment/Environments/Environment data/theta_old_env.pkl', 'rb') as f: 
    theta = pickle.load(f)
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Programms 21_04 old environment/Environments/Environment data/excel_data_old_env.pkl', 'rb') as f: 
    excel_data = pickle.load(f)
#create environment
own_env = con_env.ExtractionEnv_con(wanted_drop_size, theta, excel_data, reward_weights)

#%%

class myPID:
    dt  = 0.0
    max = 0.0
    min = 0.0
    kp  = 0.0
    kd  = 0.0
    ki  = 0.0
    err = 0.0
    int = 0.0
    def __init__(self, dt, max, min, kp, kd, ki) :
        self.dt  = dt
        self.max = max
        self.min = min
        self.kp  = kp
        self.kd  = kd
        self.ki  = ki
    def run(self,set,state) :
        error = set - state;
    
        P = self.kp * error
 
        self.int += error * self.dt
        I = self.ki * self.int
 
        D = self.kd * (error - self.err) / self.dt
 
        output = P + I + D
        if output > self.max :
            output = self.max
        elif output < self.min :
            output = self.min
 
        self.err = error
        return(output)
 
#%%


def main() :
    pid = myPID(1, 50, -50, 100, 50, 1)
    state = own_env.set_reset(550,30)
    print(state)
    extr_course = []
    extr_course.append(state)
    for i in range(200):
        
        rpm_change = pid.run(1, state[0])
        print(rpm_change)
        #if (state['drop_size']< 0.90) or state['drop_size']>1.10:
        state = own_env.step([-rpm_change,0])[0]
        #else:
            #state = env.set_reset(state['rpm'], state['solvent'])
        print(state)
        extr_course.append(state)
    #print('val:','{:7.3f}'.format(val),' inc:','{:7.3f}'.format(inc) )
    #val += inc
    return extr_course
#%%
extr_course = main()
#------------------------------------------------------------------------------
#%%
#test with acetone environment
con_env = SourceFileLoader("Environment_con_with_aceton_opt_rew_27_04_22.py", r"C:/Users/marvi/sciebo/Masterarbeit/Programms 11_04 new environment/Environments/Environment_con_with_aceton_opt_rew_27_04_22.py").load_module()
wanted_acetone_conc=1.2
#set own reward_weights
reward_weights = [1,0.3,0.3,0]

#load data for environment creation
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Programms 11_04 new environment/Environments/Environment data/theta.pkl', 'rb') as f: 
    theta = pickle.load(f)
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Programms 11_04 new environment/Environments/Environment data/excel_data.pkl', 'rb') as f: 
    excel_data = pickle.load(f)
#create environment
own_env = con_env.ExtractionEnv_con(wanted_acetone_conc, theta, excel_data, reward_weights)
#%%

def main_ace() :
    pid = myPID(1, 2, -2, 6, 3, 0)
    state = own_env.set_reset(400,20)
    print(state)
    extr_course = []
    extr_course.append(state)
    for i in range(50):
        print(state[4])
        solvent_change = pid.run(1.2, state[4])
        print(solvent_change)
        #if (state['drop_size']< 0.90) or state['drop_size']>1.10:
        state = own_env.step([0,-solvent_change])[0]
        #else:
            #state = env.set_reset(state['rpm'], state['solvent'])
        print(state)
        extr_course.append(state)
    #print('val:','{:7.3f}'.format(val),' inc:','{:7.3f}'.format(inc) )
    #val += inc
    return extr_course
#%%
extr_course = main_ace()


#%%
drop_size = []
rpm = []
aceton_conc = []
solvent = []
for item in extr_course:
    drop_size.append(item[0])
    rpm.append(item[2])
    aceton_conc.append(item[4])
    solvent.append(item[3])
time = np.arange(len(drop_size))+1

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Action [-]',size=16)
ax1.set_ylabel('rpm [1/min]', color=color,size=16)
ax1.plot(time, solvent, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.axis([1, 50,5,25 ])
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Drop size [mm]', color=color,size=16)  # we already handled the x-label with ax1
ax2.plot(time, drop_size, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.axis([1, 50, 0.5, 1.5])
fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.set_size_inches(25, 8)



#%%
env = environment.ExtractionEnv(wanted_size,all_theta,data_2,reward_weights)
print(env.set_reset(550,33))
