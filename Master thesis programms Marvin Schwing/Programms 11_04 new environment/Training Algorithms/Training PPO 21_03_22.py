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
import matplotlib.pyplot as plt
import seaborn as sns
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
        #print('state: {}'.format(state))
        #print('probs: {}'.format(probs))
        dist = tfp.distributions.Categorical(probs=probs)
        #print('this is dist: {}'.format(dist))
        action = dist.sample()
        #print('action: {}'.format(action))
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
            state_arr, action_arr, old_prob_arr, vals_arr, vals_arr_new, reward_arr, dones_arr, batches = self.memory.generate_batches()
            values = vals_arr
            values_next = vals_arr_new
            advantage_1 = np.zeros(len(reward_arr), dtype=np.float32)
            #print(advantage)
            #print('state_arr: {}'.format(state_arr))
            advantage = np.zeros(len(reward_arr) + 1)
            for t in reversed(range(len(reward_arr))):
                delta = reward_arr[t] + (self.gamma * values_next[t] * (1-int(dones_arr[t]))) - values[t]
                advantage[t] = delta + (self.gamma * self.gae_lambda * advantage[t + 1] * (1-int(dones_arr[t])))
            
            #print('advantage: {}'.format(advantage))
            #for t in range(len(reward_arr)-1):
            #    discount = 1
            #    a_t = 0
            #    for k in range(t, len(reward_arr)-1):
            #        a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*(1-int(dones_arr[k])) - values[k])
                    #print(a_t)
            #        discount *= self.gamma*self.gae_lambda
            #    advantage_1[t]=a_t  #why is the advantage sometimes so big
            #print('advantage_1: {}'.format(advantage_1))
            
            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])
                    #print('states from batch: {}'.format(actions))
                    probs = self.actor(states)
                    dist = tfp.distributions.Categorical(probs=probs)
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
                    #print('clipped probs :{}'.format(clipped_probs))
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
                #print('actor grads: {}'.format(actor_grads))
                #print('critic grads: {}'.format(critic_grads))
                #print(actor_grads)
                self.actor.optimizer.apply_gradients(zip(actor_grads, actor_params))
                self.critic.optimizer.apply_gradients(zip(critic_grads, critic_params))
                
        self.memory.clear_memory()
#%%
model_param = SourceFileLoader("model_param.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/model_param.py").load_module()
theta = SourceFileLoader("Theta.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/Theta.py").load_module()
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Python Module/Data_new_aceton_18_03.pkl', 'rb') as f: 
    data = pickle.load(f)
model_param = SourceFileLoader("new_model_param.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/new_model_param.py").load_module()
dis_env = SourceFileLoader("Environment_dis_with_aceton_25_03_22.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/Environment_dis_with_aceton_25_03_22.py").load_module()
data_2, dummie = model_param.model_parameter(data['drop_size'], data['rpm'], data['solvent'], data['flooding'], data['aceton_conc'])

all_theta = {}
for key in data_2.keys():
    #print(key)
    all_theta[key]=theta.Theta(data_2,1,'rpm','drop_size', key)

wanted_size=1.2
reward_weights = [0.6,0.2,0.6,1]

own_env = dis_env.ExtractionEnv_dis(wanted_size, all_theta, data_2, reward_weights)

#%%
own_env.reset()

#%%
        
if __name__ == '__main__':
    env = own_env
    #N = 256
    batch_size = 128
    n_epochs = 4
   
    agent = Agent(0.9,0.2, 0.2, n_epochs, 0.99, batch_size,(5),(500,200,100,16), (500,200,100,1), 'softmax',None, 0.0003, 0.0003)
    #def __init__(self,gamma,std, policy_clip, n_epochs, gae_lambda, batch_size,input_shape,layers_sizes_actor, 
    #             layers_sizes_critic, output_activation_actor,output_activation_critic, lr_actor, lr_critic):
    episodes = 10000
    n_steps = 0
    learn_iters = 0
    avg_score = 0 
    score_history = []
    for i in range(episodes): 
        print('episode: {}/{}'.format(i,episodes))
        obs = env.reset()
        done = False
        score = 0 
        if i%500 == 0:
            agent.save_actor()
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
            
            #print('reward: {}'.format(reward))
            norm_obs_2 = copy.copy(obs_2)
            norm_obs_2[0] /= 2
            norm_obs_2[2] /= 600
            norm_obs_2[3] /= 35
            dummie_action, new_val, dummie_prob = agent.choose_action(norm_obs_2)
            n_steps +=1
            score += reward
            if info==True:
                store_d = False
            else:
                store_d = done
            agent.store_transition(norm_obs_2,action,prob,val,new_val,reward,store_d)
            #print(norm_obs_2)
            #if n_steps % N==0:
            #    print('learning')
            #    agent.learn()
            #    learn_iters += 1
            obs = copy.copy(obs_2)
            
        if n_steps > 400:
            print('learning')
            agent.learn()
            n_steps = 0
            learn_iters += 1
        print('score: {}'.format(score))
        score_history.append(score)
        avg_score = np.mean(score_history[-50:])
        print('Average score: {}'.format(avg_score))
        
        
#%%
#model = agent.save_actor()
#%%
obs = env.set_reset(550, 25)
extr_course = []
for i in range(100):
    extr_course.append(obs)
    norm_obs = copy.copy(obs)
    norm_obs[0] /= 2
    norm_obs[2] /= 600
    norm_obs[3] /= 35
    state = tf.convert_to_tensor([norm_obs])
    #print(state)
    action  = model(state)
    #print(action)
    action = np.argmax(action)
    #print(action)
    obs, dummy_1, dummy_2, dummy_3 = env.step(action)
    
#%%
#print(extr_course)
#model = agent.save_actor()
#model.save('PPO_16_03_22.h5')

#%%
#tf.enable_v2_behavior()
tfd = tfp.distributions
normal = tfp.distributions.Normal(loc=0.0, scale=1.0)
#normal = tfp.distributions.Categorical([0.01,0.01,0.01,0.97])
gamma = tfp.distributions.Gamma(concentration=5.0, rate=1.0)
poisson = tfp.distributions.Poisson(rate=2.0)
laplace = tfp.distributions.Laplace(loc=0.0, scale=1.0)

sns.set_style(
    style='darkgrid', 
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100
#%%
n_samples = 800

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes = axes.flatten()

sns.distplot(a=normal.sample(n_samples), color=sns_c[0], rug=True, ax=axes[0])
axes[0].set(title=f'Normal Distribution')

sns.distplot(a=gamma.sample(n_samples), color=sns_c[1], rug=True, ax=axes[1])
axes[1].set(title=f'Gamma Distribution');

sns.distplot(a=poisson.sample(n_samples), color=sns_c[2], kde=False, rug=True, ax=axes[2])
axes[2].set(title='Poisson Distribution');

sns.distplot(a=laplace.sample(n_samples), color=sns_c[3], rug=True, ax=axes[3])
axes[3].set(title='Laplace Distribution')

plt.suptitle(f'Distribution Samples ({n_samples})', y=0.95);

#%%
