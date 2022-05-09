# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 12:51:48 2022

@author: marvi
"""
import numpy as np
import pickle
from importlib.machinery import SourceFileLoader
import copy 
import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import keras
from collections import deque
#load own environment Modul
dis_env = SourceFileLoader("Environment_dis_with_aceton_29_03_22.py", r"C:/Users/marvi/sciebo/Masterarbeit/Programms 11_04/Environments/Environment_dis_with_aceton_29_03_22.py").load_module()
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#physical_devices = tf.config.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
#%%

class Memory:
    def __init__(self, batch_size):
        self.states = [] 
        self.probs = []
        self.vals = []
        self.vals_new = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype = np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return np.array(self.states),np.array(self.actions),np.array(self.probs),np.array(self.vals),np.array(self.vals_new),np.array(self.rewards),np.array(self.dones),batches
    
    def store_memory(self, state, action, prob, val, val_new, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.vals_new.append(val_new)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear_memory(self):
        self.states = [] 
        self.probs = []
        self.vals = []
        self.vals_new = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
class ActorNet():
    def __init__(self,input_shape, layers_sizes, output_activation, lr):
        self.input_shape = input_shape
        self.layers_sizes = layers_sizes
        self.output_activation = output_activation
        self.lr = lr
                
    def create_actor_nn(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=self.input_shape))
        for h in self.layers_sizes[:-1]:
            model.add(tf.keras.layers.Dense(units=h, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.layers_sizes[-1], activation=self.output_activation))#activation should be tanh or softmax???? 2 outputs one for rpm one for solvent so tanh should be the right choice
        model.compile(loss = 'mse', optimizer = Adam(lr = self.lr))
        return model
    
class CriticNet():
    def __init__(self,input_shape, layers_sizes, output_activation, lr):
        self.input_shape = input_shape
        self.layers_sizes = layers_sizes
        self.output_activation = output_activation
        self.lr = lr
        
    def create_critic_nn(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=self.input_shape))
        for h in self.layers_sizes[:-1]:
            model.add(tf.keras.layers.Dense(units=h, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.layers_sizes[-1], activation=self.output_activation))
        model.compile(loss = 'mse', optimizer = Adam(lr = self.lr))
        return model
    
class Agent:
    def __init__(self,gamma,std, policy_clip, n_epochs, gae_lambda, batch_size,input_shape,layers_sizes_actor, 
                 layers_sizes_critic, output_activation_actor,output_activation_critic, lr_actor, lr_critic):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda # generalize advantage estimisation
        self.std = std
        self.actor = ActorNet(input_shape, layers_sizes_actor, output_activation_actor, lr_actor).create_actor_nn()
        self.critic = CriticNet(input_shape,layers_sizes_critic, output_activation_critic, lr_actor).create_critic_nn()
        #self.action_max = action_max
        self.memory = Memory(batch_size)
        
    def store_transition(self,state,action,prob,val,new_val,reward,done):
        self.memory.store_memory(state,action,prob,val,new_val,reward,done)
    
    def save_actor(self):
        self.actor.save(r'C:/Users/marvi/sciebo/Masterarbeit/Model/PPO'+ datetime.datetime.today().strftime('%Y_%m_%d %H_%M')+'.h5')
        
    def forward_actor(self, state):
        dist = self.actor(state)
        return dist
        
    def forward_critic(self, state):
        val = self.critic(state)
        return val
        
    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        probs = self.actor(state)
        dist = tfp.distributions.Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)
        return action.numpy()[0], value.numpy()[0], log_prob.numpy()[0]

    def test_action(self,observation):
        state = tf.convert_to_tensor([observation])
        probs = self.actor(state)
        action = np.argmax(probs[0])
        return action
    
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, vals_arr_new, reward_arr, dones_arr, batches = self.memory.generate_batches()
            values = vals_arr
            values_next = vals_arr_new
            advantage = np.zeros(len(reward_arr) + 1)
            rtg =  np.zeros(len(reward_arr) + 1)
            for t in reversed(range(len(reward_arr))):
                delta = reward_arr[t] + (self.gamma * values_next[t] * (1-int(dones_arr[t]))) - values[t]
                advantage[t] = delta + (self.gamma * self.gae_lambda * advantage[t + 1] * (1-int(dones_arr[t])))
                rtg[t] = reward_arr[t]+(rtg[t+1])
            print(advantage)
            print(rtg)
            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])
                    probs = self.actor(states)
                    dist = tfp.distributions.Categorical(probs=probs)
                    new_probs = dist.log_prob(actions)

                    critic_value = self.critic(states)
                    critic_value = tf.squeeze(critic_value, 1)
                    
                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)
                    weighted_clipped_probs = clipped_probs*advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs, weighted_clipped_probs)#help_matrix = weighted_probs
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    returns = advantage[batch]+values[batch]
                    rtg_batch = rtg[batch]
                    #critic_loss = keras.losses.MSE(critic_value, returns)
                    critic_loss = keras.losses.MSE(critic_value, rtg_batch)

                actor_params = self.actor.trainable_variables

                critic_params = self.critic.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_grads = tape.gradient(critic_loss, critic_params)

                self.actor.optimizer.apply_gradients(zip(actor_grads, actor_params))
                self.critic.optimizer.apply_gradients(zip(critic_grads, critic_params))
                
        self.memory.clear_memory()
#%%
#set wanted acetone concentration
wanted_acetone_conc=1.2
#set own reward_weights
reward_weights = [0.6,0.2,0.6,1]
#load data for environment creation
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Programms 11_04/Environments/Environment data/theta.pkl', 'rb') as f: 
    theta = pickle.load(f)
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Programms 11_04/Environments/Environment data/excel_data.pkl', 'rb') as f: 
    excel_data = pickle.load(f)

rpm_scale = 2
own_env = dis_env.ExtractionEnv_dis(wanted_acetone_conc, theta, excel_data, reward_weights,rpm_scale)

#%%
own_env.reset()
#%%
#use same test function as in DDPG
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
        if e_return > 50:
            early_stopping.append(e_return)
            first = early_stopping[0]
            if len(early_stopping) == 5:
                stop = True
                for stopping in early_stopping:
                    if np.abs(first-stopping)>=5:
                        stop = False
                if stop == True:
                    print('Early stopping criteria was met')
        print('test return:', e_return, 'episode length:', e_length)
    return stop

if __name__ == '__main__':
    env = own_env
    test_env = env
    batch_size = 128
    n_epochs = 4
    agent = Agent(0.9,0.2, 0.2, n_epochs, 0.99, batch_size,(5),(500,200,100,16), (500,200,100,1), 'softmax',None, 0.0003, 0.0003)
    test_agent_every = 25
    learn_iters = 0
    avg_score = 0 
    n_steps = 0
    returns = []
    steps = []
    early_stopping = deque(maxlen=5)
    stop = False
    episode = 0
    while stop == False: 
        episode += 1
        print('episode: {}'.format(episode))
        obs = env.reset()
        done = False
        episode_return = 0 
        episode_length = 0
        #if i%500 == 0:
        #    agent.save_actor()
        while not done:
            norm_obs = copy.copy(obs)
            norm_obs[0]/=2
            norm_obs[2]/=600
            norm_obs[3]/=20
            norm_obs[4]/=2
            action, val, prob = agent.choose_action(norm_obs)
            obs_2, reward, done, info = env.step(action)

            norm_obs_2 = copy.copy(obs_2)
            norm_obs_2[0] /= 2
            norm_obs_2[2] /= 600
            norm_obs_2[3] /= 20
            norm_obs_2[4] /= 2
            dummie_action, new_val, dummie_prob = agent.choose_action(norm_obs_2)
            n_steps +=1
            episode_length +=1
            episode_return += reward
            if info==True:
                store_d = False
            else:
                store_d = done
            agent.store_transition(norm_obs_2,action,prob,val,new_val,reward,store_d)
            obs = copy.copy(obs_2)
            
        returns.append(episode_return)
        steps.append(episode_length)
        if (episode > 0) and (episode % test_agent_every == 0):
            stop = test_agent()
        if n_steps > 400:
            print('learning')
            agent.learn()
            n_steps = 0
            learn_iters += 1
        print('episode_return: {}'.format(episode_return))
        avg_score = np.mean(returns[-50:])
        print('Average score last 50 returns: {}'.format(avg_score))
        
        
#%%
#model = agent.save_actor()
    
#%%
#print(extr_course)
#model = agent.save_actor()
#model.save('PPO_16_03_22.h5')


