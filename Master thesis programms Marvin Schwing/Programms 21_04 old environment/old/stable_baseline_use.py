# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 10:34:22 2022

@author: marvi
"""
from gym import Env
import gym
import numpy as np
from gym.spaces import Discrete, Box
import pickle
from importlib.machinery import SourceFileLoader
import random
import stable_baselines3
from stable_baselines3.common.env_checker import check_env

model_param = SourceFileLoader("model_param.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/model_param.py").load_module()
theta = SourceFileLoader("Theta.py", r"C:/Users/marvi/sciebo/Masterarbeit/Python Module/Theta.py").load_module()

wanted_size=1
reward_weights = [0.6,0.2,0.2,1]

class ExtractionEnv(Env):
    #reward_weights = [drop_size_weight, rpm_weight, solvent_weight, action_weight]
    def __init__(self, wanted_size, omega, data, reward_weights):
        
        #parameter for reward calculation
        #self.flooding_time = 0
        #action space in diskret umbauen!
        #self.action_space = Discrete(16)
        self.action_space = gym.spaces.Box(low = np.array([-20,-2]), high = np.array([20,+2]), shape=(2,), dtype=np.float32)
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
        done = False
        self.state[2] += action[0]
        
        self.state[3] += int(round(action[1],0)) 
        
        # if action < 11:
        #     self.state[2] += (action-5)*2
        # if action >= 11:
        #     self.state[3] += (action-13)
            
        if (self.state[3] < 25) or (self.state[3] > 35):
            done = True
            reward = -50 -self.extract_length
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
            reward = reward - 10
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
        return np.array(self.state).astype(np.float32), reward, done, info
    
    def reset(self):
        # Reset extraction state with some randomness
        #self.drop_size = 1.2
        rpm = random.randrange(300, 550, 50)
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
        
        return np.array(self.state).astype(np.float32)
    
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
with open(r'C:/Users/marvi/sciebo/Masterarbeit/Python Module/Data_3.pkl', 'rb') as f: #try could be useful
    data = pickle.load(f)
    
data_2, dummie = model_param.model_parameter(data['drop_size'], data['rpm'], data['solvent'], data['flooding'])

all_theta = {}
for key in data_2.keys():
    #print(key)
    all_theta[key]=theta.Theta(data_2,1,'rpm','drop_size', key)  
    
own_env = ExtractionEnv(wanted_size, all_theta, data_2, [0.6,0.2,0.2,1])
#%%
check_env(own_env, warn=True)

#%%
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo.policies import MlpPolicy

#%%

model = PPO(MlpPolicy, own_env, verbose=1)
#%%
def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = own_env
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        print(obs)
        step = 1
        while not done:
            
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            print(action)
            print(obs)
            #print(done)
            episode_rewards.append(reward)
            step += 1
        print(step)
        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward
#%%
mean_reward_before_train = evaluate(model, num_episodes=10)

#%%

from stable_baselines3.common.evaluation import evaluate_policy
mean_reward, std_reward = evaluate_policy(model, own_env, n_eval_episodes=10)
#model =  PPO(MlpPolicy, own_env, verbose=1)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
new_model = model.learn(total_timesteps=1000000)
#model.save("ppo_1")
mean_reward, std_reward = evaluate_policy(new_model, own_env, n_eval_episodes=10)
#del model
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

#%%
#model = PPO.load("ppo_1")
mean_reward_after_train = evaluate(new_model, num_episodes=1)
