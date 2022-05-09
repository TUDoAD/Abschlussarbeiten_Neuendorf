# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:20:10 2022
https://towardsdatascience.com/deep-deterministic-policy-gradient-ddpg-theory-and-implementation-747a3010e82f
@author: marvi
"""
import numpy as np
import pickle
from importlib.machinery import SourceFileLoader
import tensorflow as tf
import datetime
import copy
from collections import deque
#load own environment Modul
con_env = SourceFileLoader("Environment_con_with_aceton_28_03_22.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/Environment_con_with_aceton_28_03_22.py").load_module()
#load buffer for replay memory
buffer = SourceFileLoader("buffer.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/buffer.py").load_module()
#%%
#set wanted acetone concentration
wanted_acetone_conc=1.2
#set own reward_weights
reward_weights = [1,0.4,0.4,1]

#load data for environment creation
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Programms 11_04/Environments/Environment data/theta.pkl', 'rb') as f: 
    theta = pickle.load(f)
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Programms 11_04/Environments/Environment data/excel_data.pkl', 'rb') as f: 
    excel_data = pickle.load(f)
#create environment
own_env = con_env.ExtractionEnv_con(wanted_acetone_conc, theta, excel_data, reward_weights)
#create buffer
BasicBuffer_b = buffer.BasicBuffer_b
#%%
#function to generate neuronal networks
def ANN2(input_shape,layer_sizes, hidden_activation='relu', output_activation=None):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))    
    for h in layer_sizes[:-1]:
        model.add(tf.keras.layers.Dense(units=h, activation='relu'))
    model.add(tf.keras.layers.Dense(units=layer_sizes[-1], activation=output_activation))
    model.compile
    return model
#DDPG Algorithm in a function
def ddpg(env_fn,
         test_agent_every=25,
         replay_size=int(1e5),
         gamma=0.99, 
         decay=0.99,
         actor_lr=1e-3,
         critic_lr=1e-3,
         batch_size=128,
         start_steps=2000, 
         action_noise=[0.05,0.05],
         max_episode_length=200):
    
    pushed_transitions = np.array([0,0,0,0]) #just shows how balanced the actions are in the replay buffer
    early_stopping = deque(maxlen=5) # for saving the last 5 cummulated rewards (returns) of the testing sequence
    stop = False
    env, test_env = env_fn, env_fn
    num_states = 5
    num_actions = 2
    
    #define the shape of the neural nets
    s_shape = (num_states)
    sA_shape = (num_states+num_actions)
    hidden_sizes_1=(50,100,150) #hidden layers actor net
    hidden_sizes_2=(400,300,200) #hidden layers critic net
    
    action_max = env.action_space.high
    #Initialize Main Networks
    actor = ANN2(s_shape,list(hidden_sizes_1)+[num_actions], hidden_activation='relu', output_activation='tanh')
    critic = ANN2(sA_shape, list(hidden_sizes_2)+[1], hidden_activation='relu')

    #Initialize Target Networks
    actor_target = ANN2(s_shape,list(hidden_sizes_1)+[num_actions], hidden_activation='relu', output_activation='tanh')
    critic_target = ANN2(sA_shape, list(hidden_sizes_2)+[1], hidden_activation='relu')
    #Initialize the target networks with the same weights of the main network
    actor_target.set_weights(actor.get_weights())
    critic_target.set_weights(critic.get_weights())

    #Initialize replay memory 
    replay_buffer = BasicBuffer_b(size=replay_size,obs_dim=num_states, act_dim=num_actions)

    #Train each network separately
    actor_optimizer =tf.keras.optimizers.Adam(learning_rate=actor_lr)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
    
    def get_action(new_s, noise_scale):
        a = actor.predict(np.array(new_s).reshape(1,-1))
        #noise_scale for exploration purposes
        a += noise_scale * np.random.randn(num_actions)
        #clip the action in the tanh scale between -1 and 1
        return np.array([np.clip(a[0][0], -1, 1),np.clip(a[0][1], -1, 1)])
        
    #function to test the Actor during training
    def test_agent(num_episodes=3):
        test_returns = []
        n_steps = 0
        for j in range(num_episodes):
            s, episode_return, episode_length, d = test_env.reset(), 0, 0, False
            #normalize the state 
            new_s = copy.copy(s)
            new_s[0]=new_s[0]/2
            new_s[2]=new_s[2]/600
            new_s[3]=new_s[3]/20
            new_s[4]=new_s[4]/2
            while not (d or (episode_length == max_episode_length)):
            #Take deterministic action at test time (noise_scale=0)
                del s
                #convert action Output, which is in [-1,1] to rpm output in [-20,20] and solvent output in [-2,2]
                a = action_max * get_action(new_s, 0)
                s, r, d, _ = test_env.step(a)
                del new_s
                new_s = copy.copy(s)
                new_s[0]=new_s[0]/2
                new_s[2]=new_s[2]/600
                new_s[3]=new_s[3]/20
                new_s[4]=new_s[4]/2
                episode_return += r
                episode_length += 1
                n_steps += 1
                #the stop bolean is for early stopping
                stop = False
            if episode_return > 100:
                early_stopping.append(episode_return)
                first = early_stopping[0]
                if len(early_stopping) == 5:
                    for stopping in early_stopping:
                        if np.abs(first-stopping)<=5:
                            stop = True
                        else:
                            stop = False
                    if stop == True:
                        print('Early stopping criteria was met')
            print('test return:', episode_return, 'episode_length:', episode_length)
            test_returns.append(episode_return)
        return stop
    
    
    #Main loop: play episode and train
    returns = []
    actor_losses = []
    critic_losses = []
    num_steps = 0
    stop = False
    i_episode = 0
    inbuffer = 0
    #training while early stopping is not met
    while stop == False: 
        i_episode += 1
        #reset env
        s, episode_return, episode_length, d, info = env.reset(), 0, 0, False, False
        print('starting state: {}'.format(s))
        new_s = copy.copy(s)
        new_s[0]=new_s[0]/2
        new_s[2]=new_s[2]/600
        new_s[3]=new_s[3]/20
        new_s[4]=new_s[4]/2
        stop == False
        
        #d says whether an epsiode is done or not 
        while not d:
            # For the first `start_steps` steps, use randomly sampled actions
            # in order to encourage exploration.
            if num_steps > start_steps: #after a certain amount of steps use the network to get actions
                a = action_max * get_action(new_s, action_noise)
            else: #filling the replay buffer with random actions for exploration
                a = env.action_space.sample()
              # Keep track of the number of steps done
            num_steps += 1
            if num_steps == start_steps:
                print("USING AGENT ACTIONS NOW")
            # take an action in the environment
            s2, r, d, _ = env.step(a)
            episode_return += r
            episode_length += 1
            # Ignore the "done" signal if it comes from hitting the time horizon (that is, when it's an artificial terminal signal that isn't based on the agent's state)
            d_store = False if info == True else d 
            new_s2 = copy.copy(s2)
            new_s2[0]=new_s2[0]/2
            new_s2[2]=new_s2[2]/600
            new_s2[3]=new_s2[3]/20
            new_s2[4]=new_s2[4]/2
           
            #pushed_transitions = np.array([++,+-,-+,--])
            #gives an overview over the transitions in the replay memory
            if a[0]>0 and a[1]>0:
                pushed_transitions[0]+=1
            elif a[0]>0 and a[1]<0:
                pushed_transitions[1]+=1
            elif a[0]<0 and a[1]>0:
                pushed_transitions[2]+=1
            elif a[0]<0 and a[1]<0:
                pushed_transitions[3]+=1
                
            #Store experience to replay memory
            replay_buffer.push(new_s, a/action_max, r, new_s2, d_store)
            inbuffer +=1
            s = copy.copy(s2)
            new_s = copy.copy(new_s2)
            del s2
            del new_s2
                
        #after one episode perform the update of the networks
        print('Peform Update')
        for _ in range(5):
            #checks if enough transitions are in the memory
            if inbuffer<batch_size:
                print('replay memory is too small')
                break
            #get a random sample out of the replay buffer
            S,A,R,S2,D = replay_buffer.sample(batch_size)
            S = np.asarray(S,dtype=np.float32)
            A = np.asarray(A,dtype=np.float32)
            R = np.asarray(R,dtype=np.float32)
            S2 = np.asarray(S2,dtype=np.float32)
            D = np.asarray(D,dtype=np.float32)
            #Critic optimization
            with tf.GradientTape() as tape:
                A2 = actor_target(S2)
                S2A2 = np.concatenate((S2,A2),axis=1)
                target = R + gamma *(1-D) * critic_target(S2A2)
                SA = np.concatenate((S,A),axis=1)
                Q_val = critic(SA)
                critic_loss = tf.reduce_mean((Q_val-target)**2)
                gradient_critic = tape.gradient(critic_loss, critic.trainable_variables)
            critic_losses.append(critic_loss)
            critic_optimizer.apply_gradients(zip(gradient_critic, critic.trainable_variables))
            
            #Actor optimization   
            with tf.GradientTape() as tape1:
                A_ = tf.convert_to_tensor(actor(S))
                S_ = tf.convert_to_tensor(S)
                SA_ = tf.keras.layers.concatenate([S_,A_],axis=1)
                Q = critic(SA_)
                actor_loss =  -tf.reduce_mean(Q)  #minus because gradient ascent
                gradient_actor = tape1.gradient(actor_loss,actor.trainable_variables)
            actor_losses.append(actor_loss)
            actor_optimizer.apply_gradients(zip(gradient_actor, actor.trainable_variables))
            
            #soft updating critic with polyak
            
            temp1 = np.array(critic_target.get_weights())
            temp2 = np.array(critic.get_weights())
            temp3 = decay*temp1 + (1-decay)*temp2
            critic_target.set_weights(temp3)
           
      
            #soft updating Actor network with polyak
            temp1 = np.array(actor_target.get_weights())
            temp2 = np.array(actor.get_weights())
            temp3 = decay*temp1 + (1-decay)*temp2
            actor_target.set_weights(temp3)
        
  
        print("Episode:", i_episode, "Return:", episode_return, 'episode_length:', episode_length)
        returns.append(episode_return)
        #Test the agent
        if i_episode > 0 and i_episode % test_agent_every == 0:
            stop = test_agent()
            print(pushed_transitions)
    return (returns,critic_losses,actor_losses, actor)
#%%
#start training
returns, critic_losses,actor_losses, actor = ddpg(own_env)

#%%
#manually save the actor network
actor.save(r'C:/Users/marvi/sciebo/Masterarbeit/Model/DDPG_optimised_net'+ datetime.datetime.today().strftime('%Y_%m_%d %H_%M')+'.h5')
print('network was saved')