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


model_param = SourceFileLoader("model_param.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/model_param.py").load_module()


class ExtractionEnv(Env):
    #reward_weights = [drop_size_weight, rpm_weight, solvent_weight, action_weight]
    def __init__(self, wanted_size, omega, data, reward_weights):
        
        #parameter for reward calculation
        #self.flooding_time = 0
        #action space in diskret umbauen!
        self.action_space = Discrete(16)
        #self.action_space = spaces.Box(low = np.array([-20,-2]), high = np.array([20,+2]), shape=(2,), dtype=np.float32)
        #self.observation_space = gym.spaces.Dict({'drop_size': Box(low=0, high=5, shape=(1,)),
        #        'flooding': Discrete(2), 'rpm':  Box(low=200, high=600, shape=(1,),dtype=np.float32 ), 
        #        'solvent': Discrete(60), 'feed': Discrete(60)
        #        })                                         
                                                    #[drop,flooding,rpm,solvent]
        self.observation_space = gym.spaces.Box(low = np.array([0,0,200,25]), high = np.array([2,1,600,35], dtype=np.float32))
        self.data = data
        self.omega = omega
        self.wanted_size = wanted_size
        self.reward_weights = reward_weights
        # Set start 
        # randomize the dropsize
        # calculation of the rpm in dependence of the drop size
        #self.state = dict({'drop_size': round(self.drop_size,2),
        #                   'flooding': 1, #only placeholder
        #                   'rpm': 400, #only placeholder
        #                   'solvent': 30
        #                  })
        # Set Extraction Length
        self.extract_length = 200
    def step(self, action):
        # Apply action: Change rpm or solvent flow
        reward = 0
        done= False
        if action < 11:
            self.state[2] += (action-5)*2 #kleinere änderung beim nächsten mal!!!!!
        if action >= 11:                    #25.02 rpm schritte verkleinert
            self.state[3] += (action-13)
            
        if (self.state[3] < 25) or (self.state[3] > 35):
            done = True
            reward = -50 - self.extract_length
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
        
        rpm_cost = 1-self.state[2]/600
        
        solvent_cost = 1-self.state[3]/35
        
        #if (action[0] == 0) or (int(round(action[1],0)) == 0):
        #    action_cost = 0
        #else:
        action_cost = 0#-1
        
        
        
        reward = reward + self.reward_weights[0]*(drop_size_err) + self.reward_weights[1]*rpm_cost + self.reward_weights[2]*solvent_cost + self.reward_weights[3]*action_cost
        #gewichtung testen
        
        #self.state['state'] == 0 means flooding
        if self.state[1] == 0:
            reward = reward - 100
        # Check if extraction is done
        # if drop is too small or to big: End the run
        if self.extract_length <= 0: 
            done = True
        elif (self.state[3] < 25) or (self.state[3] > 35):
            reward = reward - 50 - self.extract_length
            done = True
        elif (self.state[2] <= 300) or (self.state[2] >= 550):
            reward = reward - 50 - self.extract_length
            done = True
        
        
        # Apply state noise
        #self.state['drop_size'] += random.uniform(-0.05,0.05)
        
        # Set placeholder for info
        info = {}
         
        # Return step information
        return self.state, reward, done, info
    
    def reset(self):
        # Reset extraction state with some randomness
        #self.drop_size = 1.2
        rpm = random.randrange(300, 550, 10)
        solvent = round(random.uniform(25,35))
        
        #self.state = dict({'drop_size': 0,#only placeholder
        #                   'flooding': 0, #only placeholder
        #                   'rpm': rpm, 
        #                   'solvent': solvent
        #                  })
        self.state = [0,0,rpm,solvent]
        
        #print('solvent {}'.format(solvent))
        flooding = round(self.data['solvent {}'.format(solvent)]['flooding'][self.data['solvent {}'.format(solvent)]['rpm'].index(round(rpm/50)*50)])
        #self.drop_size = random.choice([2.5,1.5])
        Omega = self.omega['solvent {}'.format(solvent)]
        #print(Omega)
        drop_omega = 0
        for t in range(len(Omega)):
            drop_omega = drop_omega + Omega[t][0]*self.state[2]**t
        #print(drop_omega[0])
        #self.state = dict({'drop_size': round(drop_omega + random.uniform(-0.10, 0.10),2),
        #                   'flooding': flooding, 
        #                   'rpm': rpm,
        #                   'solvent': solvent
        #                  })
        self.state = [round(drop_omega + random.uniform(-0.10, 0.10),3), flooding, rpm,solvent]
        # Reset extraction time, flooding time, out of DOS time
        self.extract_length = 200
        self.flooding_time = 0
        
        return self.state
    
    def set_reset(self, start_rpm, start_solvent):
        
        
        
        rpm = start_rpm
        solvent = start_solvent
        self.state = [0,0,rpm,solvent]
        #self.state = dict({'drop_size': 0,#only placeholder
        #                   'flooding': 0, #only placeholder
        #                   'rpm': rpm, 
        #                   'solvent': solvent
        #                  })
        
        flooding = round(self.data['solvent {}'.format(solvent)]['flooding'][self.data['solvent {}'.format(solvent)]['rpm'].index(round(rpm/50)*50)])
        Omega = self.omega['solvent {}'.format(solvent)]
        drop_omega = 0
        for t in range(len(Omega)):
            drop_omega = drop_omega + Omega[t][0]*self.state[2]**t
        #print(drop_omega)
        #self.state = dict({'drop_size': round(drop_omega + random.uniform(-0.10, 0.10),2),
        #                   'flooding': flooding, 
        #                   'rpm': rpm,
        #                   'solvent': solvent
        #                  })
        self.extract_length = 200
        self.flooding_time = 0
        
        self.state = [round(drop_omega + random.uniform(-0.10, 0.10),3), flooding, rpm,solvent]
        return self.state
#%%
model_param = SourceFileLoader("model_param.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/model_param.py").load_module()
theta = SourceFileLoader("Theta.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/Theta.py").load_module()
reward_weights = [0.9,0.05,0.05,1]
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Python Module/Data_3.pkl', 'rb') as f: #try could be useful
    data = pickle.load(f)
    
data_2, dummie = model_param.model_parameter(data['drop_size'], data['rpm'], data['solvent'], data['flooding'])

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
 
#%%
class DQNAgent:
    # epsilon_greedy, epsilon_min, epsilon_decay decide if the action is random or model based
    # memory saves the namedtuple "Transition", max_memory_size gives the size of the memory
    def __init__(
            self, env, discount_factor = 0.99, epsilon_greedy = 1.0, epsilon_min=0.01, 
            epsilon_decay=0.999, learning_rate=1e-3,max_memory_size = 5000):
        self.enf = env
        self.state_size = 1
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=max_memory_size)
        self.gamma = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate
        #self._build_nn_model()
        self.num_states = 4
        self.num_actions = 16
        self.X_shape = (self.num_states)
        self.hidden_sizes_1 = (1000,500,200)
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
    
    def remember(self, transition):
        self.memory.append(transition)
        length_memory = len(self.memory)
        return length_memory
        
    def choose_action(self, state):
        new_s = copy.copy(state[0])
        new_s[0]=new_s[0]/2
        new_s[2]=new_s[2]/600
        new_s[3]=new_s[3]/35
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
            batch_states.append(s.flatten())
            batch_targets.append(target_all)
            self._adjust_epsilon()
        self.model.fit(x=np.array(batch_states),y=np.array(batch_targets),epochs = 1, verbose = 0)
        #return self.model
    
    def _adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
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
        self.model.save_weights('DQN_weights_24_02_22_1',overwrite = True)
        self.model.save('DQN_model_24_02_22_1.h5')
        
#%%
#EPISODES = 5 
batch_size = 64
init_replay_memory_size = 2000
early_stopping = deque(maxlen=5)
stop = False
Transition = namedtuple('Transition', ('state', 'action', 'reward','next_state', 'done'))
wanted_size = 1
if __name__ == '__main__':
    env = ExtractionEnv(wanted_size, all_theta, data_2, [0.6,0.2,0.2,1]) 
    testEnv = ExtractionEnv(wanted_size, all_theta, data_2, [0.6,0.2,0.2,1])
    agent = DQNAgent(env)
    state = env.reset()
    #raise SystemExit("Stop right there!") 
    state = np.reshape(state, [1, -1])
    action_in_memory = np.zeros(16)
    ## initiliaze replay buffer
    for i in range(init_replay_memory_size):
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state,[1, -1])
        new_s = copy.copy(state[0])
        new_s[0]=new_s[0]/2
        new_s[2]=new_s[2]/600
        new_s[3]=new_s[3]/35
        new_s2 = copy.copy(next_state[0])
        new_s2[0]=new_s2[0]/2
        new_s2[2]=new_s2[2]/600
        new_s2[3]=new_s2[3]/35
        agent.remember(Transition(new_s, action, reward, new_s2, done))
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
        state = env.reset()
        state = np.reshape(state, [1, -1])
        sum_reward = 0
        done = False
        i = 1
        while not done:
            action = agent.choose_action(state)
            #action_in_memory[action] += 1 
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state,[1, -1])
            new_s = copy.copy(state[0])
            new_s[0]=new_s[0]/2
            new_s[2]=new_s[2]/600
            new_s[3]=new_s[3]/35
            new_s2 = copy.copy(next_state[0])
            new_s2[0]=new_s2[0]/2
            new_s2[2]=new_s2[2]/600
            new_s2[3]=new_s2[3]/35
            #print(action_in_memory[action])
            #print(np.sum(action_in_memory)/16)
            if action_in_memory[action] < np.sum(action_in_memory)/16:
                agent.remember(Transition(new_s, action, reward, new_s2, done))
                action_in_memory[action] += 1 

            del new_s
            del new_s2
            sum_reward += reward
            state = next_state
            if i % 10 == 0:
                print('Step: {}, Reward: {}/{}'.format(i,sum_reward,5+i*5))
                
                
            if done:
                test_model = agent.get_model()
                episodes = 1
                for episode in range(1, episodes+1):
                    teststate = testEnv.reset()
                    done = False
                    score = 0
                    while not done:
                        teststate = np.reshape(teststate, [1, -1])
                        new_s = copy.copy(teststate[0])
                        new_s[0]=new_s[0]/2
                        new_s[2]=new_s[2]/600
                        new_s[3]=new_s[3]/35
                        new_s = np.reshape(new_s, [1,-1])
                        action = test_model.predict(new_s)[0]
                        print(teststate)
                        print(action)
                        del new_s
                        #print(action)
                        action = np.argmax(action)
                        #print(action)
                        teststate, reward, done, info = testEnv.step(action)
                        #print(state)
                        score+=reward
                    if score > 0:
                        early_stopping.append(score)
                        first = early_stopping[0]
                        if len(early_stopping)==5:
                            for stopping in early_stopping:
                                if np.abs(first-stopping)<=5:
                                    stop = True
                                else:
                                    stop = False
                            if stop == True:
                                print('Early stopping criteria was met')
                                agent.save_model()
                    
                print('Actual Model peformance: {}'.format(score))
                #print(score)
                total_rewards.append(sum_reward)
                print('Episode ended after {} Steps'.format(i))
                print('Episode: %d, Total Reward: %d'
                         % (e, sum_reward))
                e += 1
            #loss = agent.replay(batch_size)
            #losses.append(loss)
            agent.replay(batch_size)
            i += 1
            print('Actions in Memory: {}'.format(action_in_memory))
            #raise SystemExit("Stop right there!")
    agent.save_model()  
        # ich glaub das falsche lernen kommt daher, dass im replaybuffer nur noch die aktionen 13 und 5 vorhanden sind,
        # da diese auch im laufe des trainings sehr oft vorkommen!
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
new_s[3]=new_s[3]/35
new_s = np.reshape(new_s, [1,-1])
print(teststate)
print(new_s)
action = test_model.predict(new_s)
print(action)
#%%
test_model.save_weights('DQN_weights_25_02_22_1',overwrite = True)
test_model.save('DQN_model_25_02_22_1.h5', overwrite = True)