# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:20:10 2022

@author: marvi
"""
import numpy as np
import pickle
from importlib.machinery import SourceFileLoader
import random
import tensorflow as tf
import copy
from tensorflow.keras.optimizers import Adam
from collections import namedtuple
from collections import deque
import matplotlib.pyplot as plt
import datetime
dis_env = SourceFileLoader("Environment_dis_with_aceton_29_03_22.py", r"C:/Users/marvi/sciebo/Masterarbeit/Programms 11_04 new environment/Environments/Environment_dis_with_aceton_29_03_22.py").load_module()


#%%
#set wanted acetone concentration
wanted_acetone_conc=1.2
#set own reward_weights
reward_weights = [1, 0.3, 0.3, 0]
#load data for environment creation
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Programms 11_04 new environment/Environments/Environment data/theta.pkl', 'rb') as f: 
    theta = pickle.load(f)
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Programms 11_04 new environment/Environments/Environment data/excel_data.pkl', 'rb') as f: 
    excel_data = pickle.load(f)




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
                    del self.memory[len(self.memory)-1-i]
                    break
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
    
    def test_action(self,state):
        new_s = np.reshape(state, [1,-1])
        q_values = self.model.predict(new_s)[0]
        return np.argmax(q_values)
    
    def _learn(self, batch_samples):   # rewards are clipped to -1-1 for stabilization reasons: Henderson et al. 2018 An Introduction to deep reinforcement learning
        batch_states, batch_targets = [], []
        for transition in batch_samples:
            s, a, r, next_s, done = transition
            if done:
                target = r
            else:
                next_s = np.reshape(next_s,[1, -1])
                target = (r + self.gamma * np.amax(self.model.predict(next_s)[0]))
            s = np.reshape(s,[1, -1])
            target_all = self.model.predict(s)[0]
            target_all[a] = target
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
        self._learn(samples)
        #return history.history['loss'][0]
    
    def get_model(self):
        return self.model
    
    def save_model(self):
        self.model.save(r'C:/Users/marvi/sciebo/Masterarbeit/Model/DQN'+ datetime.datetime.today().strftime('%Y_%m_%d %H_%M')+'.h5')
        
#%%
#EPISODES = 5 
def test_agent(num_episodes=3):
    n_steps = 0
    for j in range(num_episodes):
        s, e_return, e_length, d = test_env.reset(), 0, 0, False
        #normalize the state 
        new_s = copy.copy(s)
        new_s[0]=new_s[0]/2
        new_s[2]=new_s[2]/600
        new_s[3]=new_s[3]/20
        new_s[4]=new_s[4]/2
        while not d:
            del s
            action = agent.test_action(new_s)
            print('test_action: {}'.format(action))
            s, r, d, _ = test_env.step(action)
            del new_s
            new_s = copy.copy(s)
            new_s[0]=new_s[0]/2
            new_s[2]=new_s[2]/600
            new_s[3]=new_s[3]/20
            new_s[4]=new_s[4]/2
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
                if stop == True:
                    print('Early stopping criteria was met')
        print('test return:', e_return, 'episode length:', e_length)
    return stop


batch_size = 128
init_replay_memory_size = 100000
early_stopping = deque(maxlen=5)
stop = False
Transition = namedtuple('Transition', ('state', 'action', 'reward','next_state', 'done'))
wanted_acetone_conc=1.2
global returns_test
returns_test = []
global steps_test
steps_test = []



if __name__ == '__main__':
    env = dis_env.ExtractionEnv_dis(wanted_acetone_conc, theta, excel_data,  reward_weights,10) 
    test_env = dis_env.ExtractionEnv_dis(wanted_acetone_conc, theta, excel_data,  reward_weights,10)
    returns = []
    steps = []
    agent = DQNAgent(env)
    state = env.reset()
    test_agent_every = 25
    #raise SystemExit("Stop right there!") 
    state = np.reshape(state, [1, -1])
    action_in_memory = np.zeros(16)
    ## initiliaze replay buffer
    for l in range(init_replay_memory_size):
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)

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
        if info == True:
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

    losses = []
    # for early stopping
    #e = 1
    episode = 0
    print('start learning')
    while stop == False:
        state_curr = env.reset()
        print(state_curr)
        state = np.reshape(state, [1, -1])
        sum_reward = 0
        done = False
        
        episode_return = 0 
        episode_length = 0
        episode +=1
        while not done:
            action = agent.choose_action(state)
            #action_in_memory[action] += 1 
            
            next_state, reward, done, info = env.step(action)
            
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
            
            if info == True:
                d = True
            else:
                d = False
            agent.remember(Transition(new_s, action, reward, new_s2, d),action)
            
            del new_s
            del new_s2
            episode_length +=1
            episode_return += reward
            state = next_state
            #print(i)

            #if update_net % 50 == 0:
                #update_net = 0
        print('Update Neuralnet')     
        loss = agent.replay(batch_size)
        losses.append(loss)
        
                
                
        returns.append(episode_return)
        steps.append(episode_length)
        if episode % 10 == 0:
            print('episode: {}, Return: {}'.format(episode,episode_return))
        if (episode > 0) and (episode % test_agent_every == 0):
            stop = test_agent()
            #agent.replay(batch_size)
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