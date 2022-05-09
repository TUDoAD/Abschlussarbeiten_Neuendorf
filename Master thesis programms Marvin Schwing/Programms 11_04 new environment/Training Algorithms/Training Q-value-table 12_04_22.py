# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 19:46:48 2022

@author: marvi
"""
import pandas as pd
import numpy as np
import random
from collections import deque
import datetime
import time
from importlib.machinery import SourceFileLoader
import pickle
dis_env = SourceFileLoader("Environment_dis_with_aceton_29_03_22.py", r"C:/Users/marvi/sciebo/Masterarbeit/Programms 11_04 new environment/Environments/Environment_dis_with_aceton_29_03_22.py").load_module()

#%%
#set wanted acetone concentration
wanted_acetone_conc=1.2
#set own reward_weights
reward_weights = [1,0.3,0.3,0.5]
#load data for environment creation
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Programms 11_04 new environment/Environments/Environment data/theta.pkl', 'rb') as f: 
    theta = pickle.load(f)
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Programms 11_04 new environment/Environments/Environment data/excel_data.pkl', 'rb') as f: 
    excel_data = pickle.load(f)
rpm_scale = 10
own_env_dis = dis_env.ExtractionEnv_dis(wanted_acetone_conc, theta, excel_data, reward_weights,rpm_scale)
test_env = dis_env.ExtractionEnv_dis(wanted_acetone_conc, theta, excel_data, reward_weights,rpm_scale)
#%%


def Offline_Training(episodes, discount, learning_rate, env):
    NUM_ACTIONS = env.action_space.n
    test_agent_every = 25
    gamma = discount # discount factor
    alpha = learning_rate # learning rate
    epsilon = 1
    early_stopping = deque(maxlen=5)
    obs = env.reset()
    obs_round = np.around(obs,decimals=2)
    extr_df = pd.DataFrame([obs_round], index = ['state 1'])
    extr_df['Q_value'] = ''
    extr_df.at['state 1', 'Q_value'] = np.zeros([16])
    state_frequency = {'state 1': 1}
    episode = 0
    returns = []
    steps = []
    global returns_test
    returns_test = []
    global steps_test
    steps_test = []
    stop = False
    
    #--------test function--------------
    def test_agent(extr_df):
        num_episodes=3
        n_steps = 0
        for j in range(num_episodes):
            start = time.time()
            s, e_return, e_length, d = test_env.reset(), 0, 0, False
            #normalize the state 
            while not d:
            #Take deterministic action at test time (noise_scale=0)
                #convert action Output, which is in [-1,1] to rpm output in [-20,20] and solvent output in [-2,2]
                obs_round = np.around(s,decimals=2)
                state = extr_df.index[(extr_df[0]==obs_round[0])&(extr_df[1]==obs_round[1])&(extr_df[2]==obs_round[2])&(extr_df[3]==obs_round[3])&(extr_df[4]==obs_round[4])].tolist()
                #print('state in extr_df: {} / {}'.format(state,obs_round))
                if state == []:
                    extr_df_2 = pd.DataFrame([obs_round], index = ['state {}'.format(len(extr_df.index)+1)])
                    extr_df = pd.concat([extr_df, extr_df_2])
                    extr_df.at['state {}'.format(len(extr_df.index)), 'Q_value'] = np.zeros([16])
                    state = ['state {}'.format(len(extr_df.index))]
                    state_frequency[state[0]] = 1
                    
                action = np.argmax(extr_df.loc[state,'Q_value'][0])
                del s
                del obs_round
                s, r, d, _ = test_env.step(action)
                e_return += r
                e_length += 1
                n_steps += 1
                #the stop bolean is for early stopping
            stop = False
            returns_test.append(e_return)
            steps_test.append(e_length)
            if e_return > 0:
                early_stopping.append(e_return)
                first = early_stopping[0]
                if len(early_stopping) == 5:
                    stop = True
                    for stopping in early_stopping:
                        if stopping<100:
                            stop = False
                        if np.abs(first-stopping)>=5:
                            stop = False
                        #if stopping>250:
                        #    stop = True
                    if stop == True:
                        print('Early stopping criteria was met')
            end = time.time()
            print('time for one testing sequence: {}'.format(end-start))
            print('test return:', e_return, 'episode length:', e_length)
        return stop, extr_df
    #-----------------------------------
    #----------trainings loop-----------
    while stop == False:
        episode += 1
        done = False
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
        state_frequency[state[0]] += 1
        episode_return = 0 
        episode_length = 0
        start = time.time()
        while done != True:
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
                    
                obs_round = obs_round.copy()
                obs2, rew, done, info = env.step(action) #take the action
                obs2_round =  np.around(obs2,decimals=2)
                new_state = extr_df.index[(extr_df[0]==obs2_round[0])&(extr_df[1]==obs2_round[1])&(extr_df[2]==obs2_round[2])&(extr_df[3]==obs2_round[3])&(extr_df[4]==obs2_round[4])].tolist()
                if new_state == []:
                    extr_df_2 = pd.DataFrame([obs2_round], index = ['state {}'.format(len(extr_df.index)+1)])
                    extr_df = pd.concat([extr_df, extr_df_2])
                    extr_df.at['state {}'.format(len(extr_df.index)), 'Q_value'] = np.zeros([16])
                    new_state = ['state {}'.format(len(extr_df.index))]
                    state_frequency[new_state[0]] = 1
                state_frequency[new_state[0]] += 1
                extr_df.loc[state,'Q_value'][0][action] = extr_df.loc[state,'Q_value'][0][action] + alpha * (rew + gamma * np.max(extr_df.loc[new_state,'Q_value'][0])- extr_df.loc[state,'Q_value'][0][action])
                state = new_state.copy()
                obs_round = obs2_round.copy()
                episode_length +=1
                episode_return += rew
                      
        end = time.time()
        print('time for one episode: {}'.format(end - start))
        returns.append(episode_return)
        steps.append(episode_length)
        if (episode > 0) and (episode % test_agent_every == 0):
            stop, extr_df = test_agent(extr_df)
            
            #print('Episode {} Total Reward: {}'.format(episode,rew_tot))
            #print('Episode took: {} steps'.format(timez))
            #print('Average Reward for last 100 Episodes: {}'.format(rew_avg_100/100))
            print('Average Steps for last 25 Episodes: {}'.format(np.mean(steps[-25:])))
            #print('Epsilon: {}'.format(epsilon))
            #print('Time for 100 Episodes: {:.1f} s'.format(end-start))
            print('epsilon: {}'.format(epsilon))
            print('Offline Training: Episode: {}'.format(episode))
            
    return returns, steps, extr_df



#%%
start_complete_training = time.time()

reward_plot, steps_plot, extr_df =  Offline_Training(5000, 0.99, 0.99, own_env_dis)
end_complete_training = time.time()
print(end_complete_training - start_complete_training)
#%%
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Model/Q_table_new_env'+ datetime.datetime.today().strftime('%Y_%m_%d %H_%M')+'.pkl', 'wb') as f:
    pickle.dump(extr_df, f)
#%%
print(reward_plot)
print(steps_plot)