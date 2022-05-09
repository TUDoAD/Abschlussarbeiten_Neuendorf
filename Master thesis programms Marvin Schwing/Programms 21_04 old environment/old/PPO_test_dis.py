# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 12:51:48 2022

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
import copy 
class ExtractionEnv_dis(Env):
    #reward_weights = [drop_size_weight, rpm_weight, solvent_weight, action_weight]
    def __init__(self, wanted_size, omega, data, reward_weights):
        
        #parameter for reward calculation
        #self.flooding_time = 0
        #action space in diskret umbauen!
        self.action_space = gym.spaces.Discrete(16)
        #self.action_space = spaces.Box(low = np.array([-20,-2]), high = np.array([20,+2]), shape=(2,), dtype=np.float32)                                        
                                                    #[drop,flooding,rpm,solvent]
        self.observation_space = gym.spaces.Box(low = np.array([0,0,200,25]), high = np.array([2,1,600,35], dtype=np.float32))
        self.data = data
        self.omega = omega
        self.wanted_size = wanted_size
        self.reward_weights = reward_weights
        # Set start 
        # randomize the dropsize
        # calculation of the rpm in dependence of the drop size
        # Set Extraction Length
        self.extract_length = 200
    def step(self, action):
        # Apply action: Change rpm or solvent flow
        # 28.02.22: Change of the reward: should be between -1 and 1
        reward = 0
        done= False
        if action < 11:
            self.state[2] += (action-5)*2 #kleinere änderung beim nächsten mal!!!!!
        if action >= 11:                    #25.02 rpm schritte verkleinert
            self.state[3] += (action-13)
            
        if (self.state[3] < 25) or (self.state[3] > 35):
            done = True
            reward = -1
            #reward = -50 - self.extract_length
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
        
        rpm_cost = 1-((self.state[2]-300)*(1/300))
        
        solvent_cost = 1-((self.state[3]-25)*0.1)
        
        #if (action[0] == 0) or (int(round(action[1],0)) == 0):
        #    action_cost = 0
        #else:
        action_cost = 0#-1
        
        
        
        reward = reward + self.reward_weights[0]*(drop_size_err) + self.reward_weights[1]*rpm_cost + self.reward_weights[2]*solvent_cost + self.reward_weights[3]*action_cost
        #gewichtung testen
        
        #self.state['state'] == 0 means flooding
        if self.state[1] == 0:
            reward = -1
            #reward = reward - 100
        # Check if extraction is done
        # if drop is too small or to big: End the run
        if self.extract_length <= 0: 
            done = True
        elif (self.state[3] < 25) or (self.state[3] > 35):
            #reward = reward - 50 - self.extract_length
            reward = -1
            done = True
        elif (self.state[2] <= 300) or (self.state[2] >= 550):
            #reward = reward - 50 - self.extract_length
            reward = -1
            done = True
        
        
        # Apply state noise
        #self.state['drop_size'] += random.uniform(-0.05,0.05)
        
        # Set placeholder for info
        if reward == -1:
            info = True
        else:
            info = False
         
        # Return step information
        return np.array(self.state).astype(np.float32), reward, done, info
    
    def reset(self):
        # Reset extraction state with some randomness
        #self.drop_size = 1.2
        rpm = random.randrange(300, 550, 10)
        solvent = round(random.uniform(25,35))

        self.state = [0,0,rpm,solvent]
        
        #print('solvent {}'.format(solvent))
        flooding = round(self.data['solvent {}'.format(solvent)]['flooding'][self.data['solvent {}'.format(solvent)]['rpm'].index(round(rpm/50)*50)])
        #self.drop_size = random.choice([2.5,1.5])
        Omega = self.omega['solvent {}'.format(solvent)]
        #print(Omega)
        drop_omega = 0
        for t in range(len(Omega)):
            drop_omega = drop_omega + Omega[t][0]*self.state[2]**t

        self.state = [round(drop_omega + random.uniform(-0.10, 0.10),3), flooding, rpm,solvent]
        # Reset extraction time, flooding time, out of DOS time
        self.extract_length = 200
        self.flooding_time = 0
        
        return np.array(self.state).astype(np.float32)
    
    def set_reset(self, start_rpm, start_solvent):
        
        
        
        rpm = start_rpm
        solvent = start_solvent
        self.state = [0,0,rpm,solvent]
        
        flooding = round(self.data['solvent {}'.format(solvent)]['flooding'][self.data['solvent {}'.format(solvent)]['rpm'].index(round(rpm/50)*50)])
        Omega = self.omega['solvent {}'.format(solvent)]
        drop_omega = 0
        for t in range(len(Omega)):
            drop_omega = drop_omega + Omega[t][0]*self.state[2]**t

        self.extract_length = 200
        self.flooding_time = 0
        
        self.state = [round(drop_omega + random.uniform(-0.10, 0.10),3), flooding, rpm,solvent]
        return np.array(self.state).astype(np.float32)

#%%
#import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import keras


class Memory:
    def __init__(self, batch_size):
        self.states = [] 
        self.probs = []
        self.vals = []
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
        
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches
                
    def store_memory(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear_memory(self):
        self.states = [] 
        self.probs = []
        self.vals = []
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
        
    def store_transition(self,state,action,prob,val,reward,done):
        self.memory.store_memory(state,action,prob,val,reward,done)
        
        
    def forward_actor(self, state):
        dist = self.actor(state)
        return dist
        
    def forward_critic(self, state):
        val = self.critic(state)
        return val
        
    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        probs = self.actor(state)
        dist = tfp.distributions.Categorical(probs)
        action = dist.sample()
        #mean = self.actor(state)
        #value = self.critic(state)
        #dist = self.action_max * prob
        #log_prob = action.log_prob(action)
        #dist = tfp.distributions.Normal(loc=mean, scale= self.std)
        #action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)
        #multivariatenormal distribution to get log_prob!!!
        #source: https://github.com/ericyangyu/PPO-for-Beginners/blob/5cf398382a2d91848f566a96df088cf2c01a04e7/ppo.py#L260
        #https://www.youtube.com/watch?v=JjB58InuTqM
        
        return action.numpy()[0], value.numpy()[0], log_prob.numpy()[0]
        
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            
            for t in range(len(reward_arr-1)):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*(1-int(dones_arr[k])) - values[k])
                    #print(a_t)
                    discount *= self.gamma*self.gae_lambda
                    
                #print(a_t)
                advantage[t]=a_t  #why is the advantage sometimes so big
                
            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])
                    
                    probs = self.actor(states)
                    dist = tfp.distributions.Categorical(probs)
                    new_probs = dist.log_prob(actions)
                    
                    #means = self.actor(states)
                    #dists = tfp.distributions.Normal(loc=means, scale= self.std)
                    #actions = dists.sample()
                    #new_probs = dists.log_prob(actions)
                    
                    critic_value = self.critic(states)
                    critic_value = tf.squeeze(critic_value, 1) #was macht diese zeile???
                    
                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    #print(advantage[batch])
                    #print(prob_ratio)
                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)
                    weighted_clipped_probs = clipped_probs*advantage[batch]
                    #weighted_probs = np.zeros(prob_ratio.shape)
                    #print(len(advantage[batch]))
                    #for i in range(len(advantage[batch])):
                       #print(i)
                     #   weighted_probs[i]=prob_ratio[i]*advantage[i]
                        #print(help_matrix)
                    #print(help_matrix)
                    #weighted_probs = tf.convert_to_tensor(weighted_probs)
                    #weighted_probs = prob_ratio*advantage[batch]
                    #clipped_probs = tf.clip_by_value(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)
                    #print(clipped_probs) 
                    #weighted_clipped_probs = np.zeros(clipped_probs.shape)
                    #print(len(advantage[batch]))
                    #for i in range(len(advantage[batch])):
                       #print(i)
                    #    weighted_clipped_probs[i]=clipped_probs[i]*advantage[i]
                        #print(help_matrix)
                    #print(help_matrix)
                    #weighted_clipped_probs = tf.convert_to_tensor(weighted_clipped_probs)
                    #weighted_clipped_probs = clipped_probs*advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs, weighted_clipped_probs)#help_matrix = weighted_probs
                    actor_loss = tf.math.reduce_mean(actor_loss)
                    #print('actor loss:')
                    #print(actor_loss)
                    returns = advantage[batch]+values[batch]
                    critic_loss = keras.losses.MSE(critic_value, returns)
                    #print('critic_loss')
                    #print(critic_loss)
                #print(actor_loss)
                actor_params = self.actor.trainable_variables
                #print('actor params')
                #print(actor_params)
                #print('actor params')
                critic_params = self.critic.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_grads = tape.gradient(critic_loss, critic_params)
                #print('actor grads:')
                #print(actor_grads)
                self.actor.optimizer.apply_gradients(zip(actor_grads, actor_params))
                self.critic.optimizer.apply_gradients(zip(critic_grads, critic_params))
                
        self.memory.clear_memory()
        
#%%
model_param = SourceFileLoader("model_param.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/model_param.py").load_module()
theta = SourceFileLoader("Theta.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/Theta.py").load_module()

wanted_size=1
reward_weights = [0.6,0.2,0.2,1]
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Python Module/Data_3.pkl', 'rb') as f: #try could be useful
    data = pickle.load(f)
    
data_2, dummie = model_param.model_parameter(data['drop_size'], data['rpm'], data['solvent'], data['flooding'])

all_theta = {}
for key in data_2.keys():
    #print(key)
    all_theta[key]=theta.Theta(data_2,1,'rpm','drop_size', key)  
    
own_env = ExtractionEnv_dis(wanted_size, all_theta, data_2, [0.6,0.2,0.2,1])

#%%
        
if __name__ == '__main__':
    env = own_env
    N = 256
    batch_size = 64
    n_epochs = 10
   
    agent = Agent(0.9,0.2, 0.2, n_epochs, 0.99, batch_size,(4),(1000,500,200,16), (1000,500,200,1), 'softmax',None, 0.0003, 0.0003)
    #def __init__(self,gamma,std, policy_clip, n_epochs, gae_lambda, batch_size,input_shape,layers_sizes_actor, 
    #             layers_sizes_critic, output_activation_actor,output_activation_critic, lr_actor, lr_critic):
    episodes = 100000
    n_steps = 0
    learn_iters = 0
    avg_score = 0 
    score_history = []
    for i in range(episodes): 
        print('episode: {}/{}'.format(i,episodes))
        obs = env.reset()
        done = False
        score = 0 
        while not done:
            norm_obs = copy.copy(obs)
            norm_obs[0] /= 2
            norm_obs[2] /= 600
            norm_obs[3] /= 35
            #print(norm_obs)
            #raise SystemExit("Stop right there!")
            action, val, prob = agent.choose_action(norm_obs)
            #print(obs)
            #print('action')
            #print(action)
            #print(action[0])
            
            obs_2, reward, done, info = env.step(action)
            norm_obs_2 = copy.copy(obs_2)
            norm_obs_2[0] /= 2
            norm_obs_2[2] /= 600
            norm_obs_2[3] /= 35
            
            n_steps +=1
            score += reward
            if info==True:
                store_d = False
            else:
                store_d = done
            agent.store_transition(norm_obs_2,action,prob,val,reward,store_d)
            #print(norm_obs_2)
            #if n_steps % N==0:
            #    print('learning')
            #    agent.learn()
            #    learn_iters += 1
            obs = copy.copy(obs_2)
        
        if n_steps > 64:
            print('learning')
            agent.learn()
            n_steps = 0
            learn_iters += 1
        print('score: {}'.format(score))
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print('Average score: {}'.format(avg_score))
        
        
        