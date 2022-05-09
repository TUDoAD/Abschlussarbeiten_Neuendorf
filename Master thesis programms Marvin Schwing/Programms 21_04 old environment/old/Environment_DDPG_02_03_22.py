# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:20:10 2022
https://towardsdatascience.com/deep-deterministic-policy-gradient-ddpg-theory-and-implementation-747a3010e82f
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


class ExtractionEnv_con(Env):
    #reward_weights = [drop_size_weight, rpm_weight, solvent_weight, action_weight]
    def __init__(self, wanted_size, omega, data, reward_weights):
        
        #parameter for reward calculation
        #self.flooding_time = 0
        #action space in diskret umbauen!
        self.action_space = gym.spaces.Box(low = np.array([-20,-2]), high = np.array([20,+2]), shape=(2,), dtype=np.float32)
                                     
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
        self.state[2] += action[0]
       
        self.state[3] += int(round(action[1],0)) 
            
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
            done = True
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
        info = {}
         
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
model_param = SourceFileLoader("model_param.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/model_param.py").load_module()
theta = SourceFileLoader("Theta.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/Theta.py").load_module()

wanted_size=1
reward_weights = [0.6,0.2,0.6,1]
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Python Module/Data_3.pkl', 'rb') as f: #try could be useful
    data = pickle.load(f)
    
data_2, dummie = model_param.model_parameter(data['drop_size'], data['rpm'], data['solvent'], data['flooding'])

all_theta = {}
for key in data_2.keys():
    #print(key)
    all_theta[key]=theta.Theta(data_2,1,'rpm','drop_size', key)  
    
own_env = ExtractionEnv_con(wanted_size, all_theta, data_2, [0.6,0.2,0.6,1])

#%%
import tensorflow as tf
import datetime
import copy
from collections import deque
buffer = SourceFileLoader("buffer.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/buffer.py").load_module()
 
#%%
BasicBuffer_a = buffer.BasicBuffer_a
BasicBuffer_b = buffer.BasicBuffer_b
def ANN2(input_shape,layer_sizes, hidden_activation='relu', output_activation=None):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))    
    for h in layer_sizes[:-1]:
        model.add(tf.keras.layers.Dense(units=h, activation='relu'))
    model.add(tf.keras.layers.Dense(units=layer_sizes[-1], activation=output_activation))
    return model
 
def ddpg(env_fn,
         ac_kwargs=dict(),
         seed=0,
         save_folder=None,
         #num_train_episodes=5000,
         test_agent_every=25,
         replay_size=int(1e5),
         gamma=0.99, 
         decay=0.99,
         mu_lr=1e-3,
         q_lr=1e-3,
         batch_size=128, #before 32 change to 200 15.2.22
         start_steps=1000, 
         action_noise=[0.1,0.1],
         max_episode_length=200):
    
    early_stopping = deque(maxlen=5)
    stop = False
    tf.random.set_seed(seed)
    np.random.seed(seed)
    env, test_env = env_fn, env_fn
    num_states = 4
    num_actions = 2
    X_shape = (num_states)
    QA_shape = (num_actions+num_states)
   # hidden_sizes_1=(1000,500,200)
    hidden_sizes_1=(500,200,100)
    hidden_sizes_2=(400,200)
    action_max = env.action_space.high
    # Main network outputs
    mu = ANN2(X_shape,list(hidden_sizes_1)+[num_actions], hidden_activation='relu', output_activation='tanh')
    q_mu = ANN2(QA_shape, list(hidden_sizes_2)+[1], hidden_activation='relu')

    # Target networks
    mu_target = ANN2(X_shape,list(hidden_sizes_1)+[num_actions], hidden_activation='relu', output_activation='tanh')
    q_mu_target = ANN2(QA_shape, list(hidden_sizes_2)+[1], hidden_activation='relu')
    
    mu_target.set_weights(mu.get_weights())
    q_mu_target.set_weights(q_mu.get_weights())


     # Experience replay memory
    replay_buffer = BasicBuffer_b(size=replay_size,obs_dim=num_states, act_dim=num_actions)


     # Train each network separately
    mu_optimizer =tf.keras.optimizers.Adam(learning_rate=mu_lr) #maybe anders. Fehler beim speichern:WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    q_optimizer = tf.keras.optimizers.Adam(learning_rate=q_lr)
    #alternative: mu_
    
    def get_action(new_s, noise_scale):
        #normalize s
        
        #print(s)
        #a = action_max * mu.predict(np.array(new_s).reshape(1,-1))
        a = mu.predict(np.array(new_s).reshape(1,-1))
        #print(mu.predict(np.array(s).reshape(1,-1)))
        #print(a[0])
        a += noise_scale * np.random.randn(num_actions)
        #print(a[0])
        return np.array([np.clip(a[0][0], -1, 1),np.clip(a[0][1], -1, 1)])
        
    test_returns = []
    def test_agent(num_episodes=3):
        #t0 = datetime.now()
        n_steps = 0
        for j in range(num_episodes):
            s, episode_return, episode_length, d = test_env.reset(), 0, 0, False
            new_s = copy.copy(s)
            new_s[0]=new_s[0]/2
            new_s[2]=new_s[2]/600
            new_s[3]=new_s[3]/35
            
            while not (d or (episode_length == max_episode_length)):
            # Take deterministic actions at test time (noise_scale=0)

                #print(new_s)
                #print(get_action(new_s, 0))
                del s
                s, r, d, _ = test_env.step(get_action(new_s, 0))
                del new_s
                new_s = copy.copy(s)
                new_s[0]=new_s[0]/2
                new_s[2]=new_s[2]/600
                new_s[3]=new_s[3]/35
                episode_return += r
                episode_length += 1
                n_steps += 1
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
    q_losses = []
    mu_losses = []
    num_steps = 0
    stop = False
    i_episode = 0
    inbuffer = 0
    while stop == False:
        # reset env
        i_episode += 1
        s, episode_return, episode_length, d = env.reset(), 0, 0, False
        #print(s)
        new_s = copy.copy(s)
        new_s[0]=new_s[0]/2
        new_s[2]=new_s[2]/600
        new_s[3]=new_s[3]/35
        stop == False
        
        while not d: #or (episode_length == max_episode_length)):
            # For the first `start_steps` steps, use randomly sampled actions
            # in order to encourage exploration.
    
              
            if num_steps > start_steps:
                a = action_max * get_action(new_s, action_noise)
                print(a)
            else:
                a = env.action_space.sample()
    
              #print('action: {}'.format(int(round(a[1],0))))
              #print('action: {}'.format(a[0]))
              # Keep track of the number of steps done
            num_steps += 1
            if num_steps == start_steps:
                print("USING AGENT ACTIONS NOW")
                
      
            # Step the env
            #print(np.array(a))
            #print(a[0])
            #print(a)
            s2, r, d, _ = env.step(a)
    
            episode_return += r
            episode_length += 1
          
            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d_store = False if episode_length == 200 else d
            new_s2 = copy.copy(s2)
            new_s2[0]=new_s2[0]/2
            new_s2[2]=new_s2[2]/600
            new_s2[3]=new_s2[3]/35
        
              
              # Store experience to replay buffer
            #print(a/action_max)
            replay_buffer.push(new_s, a/action_max, r, new_s2, d_store)
            inbuffer +=1
            #print(new_s, a, r, new_s2, d_store)
            #print(replay_buffer)
            #raise SystemExit("Stop right there!")
                # Assign next state to be the current state on the next round
            s = copy.copy(s2)
            new_s = copy.copy(new_s2)
                  #print('important state here')
                  #print(s2)
                  #print(new_s)
            del s2
            del new_s2
                
              # Perform the updates
        print('Peform Update')
        for _ in range(episode_length):
              #print('get something from replaybuffer nomnomnom')
            
            if inbuffer<batch_size:
                print('replay buffer is too small')
                break
            X,A,R,X2,D = replay_buffer.sample(batch_size)
            #print(X)
            #print(A)
            X = np.asarray(X,dtype=np.float32)
            A = np.asarray(A,dtype=np.float32)
            R = np.asarray(R,dtype=np.float32)
            X2 = np.asarray(X2,dtype=np.float32)
            D = np.asarray(D,dtype=np.float32)
            Xten=tf.convert_to_tensor(X)   
            #print(X2)
            #Actor optimization   
            with tf.GradientTape() as tape2:
                #Aprime = action_max * mu(X)
                Aprime = mu(X)
                #print('Aprime: {}'.format(Aprime))
                temp = tf.keras.layers.concatenate([Xten,Aprime],axis=1)
                #print('temp: {}'.format(temp))
                Q = q_mu(temp)
                #print(Q)
                mu_loss =  -tf.reduce_mean(Q)
                grads_mu = tape2.gradient(mu_loss,mu.trainable_variables)
            mu_losses.append(mu_loss)
            mu_optimizer.apply_gradients(zip(grads_mu, mu.trainable_variables))
            
        #Critic Optimization
            with tf.GradientTape() as tape:
                #next_a = action_max * mu_target(X2)
                next_a =  mu_target(X2)
                temp = np.concatenate((X2,next_a),axis=1)
                q_target = R + gamma * (1 - D) * q_mu_target(temp)
                temp2 = np.concatenate((X,A),axis=1)
                qvals = q_mu(temp2) 
                q_loss = tf.reduce_mean((qvals - q_target)**2) #das ist der fehler der minimiert werden soll
                grads_q = tape.gradient(q_loss,q_mu.trainable_variables)# erster paramter gibt an was minimiert werden soll
                                                                        # zweiter paramter gibt an welche parameter geändert werden dürfen
                                                                        # grads_q sind die gradienten
            q_optimizer.apply_gradients(zip(grads_q, q_mu.trainable_variables))# gardienten werden auf q_mu angewandt!
            q_losses.append(q_loss)
            
            ## Updating both netwokrs
            ## updating Critic network
            
            temp1 = np.array(q_mu_target.get_weights())
            temp2 = np.array(q_mu.get_weights())
            temp3 = decay*temp1 + (1-decay)*temp2
            q_mu_target.set_weights(temp3)
           
      
             #updating Actor network
            temp1 = np.array(mu_target.get_weights())
            temp2 = np.array(mu.get_weights())
            temp3 = decay*temp1 + (1-decay)*temp2
            mu_target.set_weights(temp3)
        
  
        print("Episode:", i_episode, "Return:", episode_return, 'episode_length:', episode_length)
        returns.append(episode_return)
        # Test the agent
        if i_episode > 0 and i_episode % test_agent_every == 0:
            stop = test_agent()
    
    return (returns,q_losses,mu_losses, mu)
#%%

returns, q_losses,mu_losses, mu = ddpg(own_env)


#%%

print('network was saved')
mu.save_weights('DDPG_weights_02_03_22_1',overwrite = True)
mu.save('DDPG_model_02_03_22_1.h5')