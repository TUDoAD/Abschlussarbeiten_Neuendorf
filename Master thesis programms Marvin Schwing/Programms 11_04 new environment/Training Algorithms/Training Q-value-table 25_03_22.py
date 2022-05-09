# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 19:46:48 2022

@author: marvi
"""
import pandas as pd
import numpy as np
import time
import random
from collections import deque
import datetime
#%%
#print = sg.Print
def Offline_Training(episodes, discount, learning_rate, env):
    NUM_ACTIONS = env.action_space.n
    gamma = discount # discount factor
    alpha = learning_rate # learning rate
    epsilon = 1
    early_stopping = deque(maxlen=5)
    obs = env.reset()
    extr_df = pd.DataFrame([obs], index = ['state 1'])
    extr_df['Q_value'] = ''
    extr_df.at['state 1', 'Q_value'] = np.zeros([16])
    state_frequency = {'state 1': 1}
    start = time.time()
    rew_avg_100 = 0
    avg_steps = 0
    reward_plot = []
    steps_plot = []
    for episode in range(1,episodes):
        done = False
        rew_tot = 0 #reset of the total reward
        obs = env.reset() # reset of the environment
        obs_round = np.around(obs,decimals=2)
        state = extr_df.index[(extr_df[0]==obs_round[0])&(extr_df[1]==obs_round[1])&(extr_df[2]==obs_round[2])&(extr_df[3]==obs_round[3])&(extr_df[4]==obs_round[4])].tolist()
        #print('state in extr_df: {} / {}'.format(state,obs_round))
        if state == []:
            extr_df_2 = pd.DataFrame([obs_round], index = ['state {}'.format(len(extr_df.index)+1)])
            extr_df = pd.concat([extr_df, extr_df_2])
            extr_df.at['state {}'.format(len(extr_df.index)), 'Q_value'] = np.zeros([16])
            state = ['state {}'.format(len(extr_df.index))]
            state_frequency[state[0]] = 1
        timez = 0
        state_frequency[state[0]] += 1
        while done != True:
                timez += 1
                #choose action
                if np.random.rand() <= epsilon:
                    index = np.where(extr_df.loc[state,'Q_value'][0]==0)
                    if len(index[0]) == 0:
                        action = random.randrange(NUM_ACTIONS)
                    else:
                        action = random.choice(index[0])
                    if epsilon > 0.01:
                        epsilon *= 0.99995
                else:
                    index = np.where(extr_df.loc[state,'Q_value'][0]==0)
                    if max(extr_df.loc[state,'Q_value'][0]) == 0:
                        action = random.choice(index[0])
                    else:
                        action = np.argmax(extr_df.loc[state,'Q_value'][0])
                    
                #index = np.where(extr_df.loc[state,'Q_value'][0]==0) #vermutung: unnötig
                obs_round = obs_round.copy()
                obs2, rew, done, info = env.step(action) #take the action
                obs2_round =  np.around(obs2,decimals=2)
                new_state = extr_df.index[(extr_df[0]==obs2_round[0])&(extr_df[1]==obs2_round[1])&(extr_df[2]==obs2_round[2])&(extr_df[3]==obs2_round[3])&(extr_df[4]==obs2_round[4])].tolist()
                #print('new state in extr_df: {} / {}'.format(new_state,obs2_round))
                if new_state == []:
                    extr_df_2 = pd.DataFrame([obs2_round], index = ['state {}'.format(len(extr_df.index)+1)])
                    extr_df = pd.concat([extr_df, extr_df_2])
                    extr_df.at['state {}'.format(len(extr_df.index)), 'Q_value'] = np.zeros([16])
                    new_state = ['state {}'.format(len(extr_df.index))]
                    state_frequency[new_state[0]] = 1
                state_frequency[new_state[0]] += 1
                extr_df.loc[state,'Q_value'][0][action] = extr_df.loc[state,'Q_value'][0][action] + alpha * (rew + gamma * np.max(extr_df.loc[new_state,'Q_value'][0])- extr_df.loc[state,'Q_value'][0][action])
                rew_tot = rew_tot + rew
                state = new_state.copy()
                obs_round = obs2_round.copy()
                
        avg_steps += timez       
        rew_avg_100 += rew_tot
        if episode % 100 == 0:
            end = time.time()
            reward_plot.append(rew_avg_100/100)
            early_stopping.append(rew_avg_100/100)
            steps_plot.append(avg_steps/100)
            first = early_stopping[0]
            if (len(early_stopping)==5) and (rew_avg_100/100>300):
                stop = True
                for stopping in early_stopping:
                    print(np.abs(first-stopping))
                    if np.abs(first-stopping)>=20:
                        stop = False
                if stop == True:
                    print('Early stopping criteria was met')
                    break
            print('average reward: {}'.format(rew_avg_100/100))
            
            #print('Episode {} Total Reward: {}'.format(episode,rew_tot))
            #print('Episode took: {} steps'.format(timez))
            #print('Average Reward for last 100 Episodes: {}'.format(rew_avg_100/100))
            print('Average Steps for last 100 Episodes: {}'.format(avg_steps/100))
            #print('Epsilon: {}'.format(epsilon))
            #print('Time for 100 Episodes: {:.1f} s'.format(end-start))
            print('epsilon: {}'.format(epsilon))
            print('Offline Training: Episode: {}/{}'.format(episode,episodes))
            time_to_print = divmod((end-start)*((episodes-episode)/100),60)
            print('Offline Training: Time remaining: {:2.0f}:{:2.0f} min'.format(round(time_to_print[0],0),round(time_to_print[1],0)))
            avg_steps = 0
            rew_avg_100 = 0
            start = time.time()
            
    return reward_plot, steps_plot, extr_df
#%%
from importlib.machinery import SourceFileLoader
import pickle

#%%
theta = SourceFileLoader("Theta.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/Theta.py").load_module()
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Python Module/Data_new_aceton_18_03.pkl', 'rb') as f: 
    data = pickle.load(f)
model_param = SourceFileLoader("new_model_param.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/new_model_param.py").load_module()
dis_env = SourceFileLoader("Environment_dis_with_aceton_29_03_22.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/Environment_dis_with_aceton_29_03_22.py").load_module()
data_2, dummie = model_param.model_parameter(data['drop_size'], data['rpm'], data['solvent'], data['flooding'], data['aceton_conc'])

all_theta = {}
for key in data_2.keys():
    #print(key)
    all_theta[key]=theta.Theta(data_2,1,'rpm','drop_size', key)
#%%
own_env_dis = dis_env.ExtractionEnv_dis(1.2, all_theta, data_2, [1,0.3,0.3,1],10)
#%%

reward_plot, steps_plot, extr_df =  Offline_Training(5000, 0.99, 0.99, own_env_dis)

#%%
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Model/Q_table'+ datetime.datetime.today().strftime('%Y_%m_%d %H_%M')+'.pkl', 'wb') as f:
    pickle.dump(extr_df, f)
#%%
print(reward_plot)
print(steps_plot)