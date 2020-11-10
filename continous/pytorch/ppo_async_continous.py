import gym
from gym.envs.registration import register
    
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import matplotlib.pyplot as plt
import numpy as np
import sys
import numpy
import time
import datetime

import ray

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
dataType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Actor_Model(nn.Module):
    def __init__(self, state_dim, action_dim, myDevice = None):
        super(Actor_Model, self).__init__()

        self.device = myDevice if myDevice != None else device
        self.nn_layer = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Tanh()
              ).float().to(self.device)
        
    def forward(self, states):
        return self.nn_layer(states)

class Critic_Model(nn.Module):
    def __init__(self, state_dim, action_dim, myDevice = None):
        super(Critic_Model, self).__init__()   
        
        self.device = myDevice if myDevice != None else device
        self.nn_layer = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
              ).float().to(device)
        
    def forward(self, states):
        return self.nn_layer(states)

class Memory(Dataset):
    def __init__(self):
        self.states         = []
        self.actions        = []        
        self.rewards        = []
        self.dones          = []     
        self.next_states    = []

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype = np.float32), np.array(self.actions[idx], dtype = np.float32), np.array([self.rewards[idx]], dtype = np.float32), \
            np.array([self.dones[idx]], dtype = np.float32), np.array(self.next_states[idx], dtype = np.float32)

    def get_all(self):
        return self.states, self.actions, self.rewards, self.dones, self.next_states
    
    def save_all(self, states, actions, rewards, dones, next_states):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.next_states = next_states
    
    def save_eps(self, state, action, reward, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]  

class Distributions():
    def __init__(self, myDevice = None):
        self.device = myDevice if myDevice != None else device
    
    def sample(self, mean, std):
        distribution    = Normal(mean, std)
        return distribution.sample().float().to(self.device)
        
    def entropy(self, mean, std):
        distribution    = Normal(mean, std)    
        return distribution.entropy().float().to(self.device)
      
    def logprob(self, mean, std, value_data):
        distribution    = Normal(mean, std)
        return distribution.log_prob(value_data).float().to(self.device)

    def kl_divergence(self, mean1, std1, mean2, std2):
        distribution1   = Normal(mean1, std1)
        distribution2   = Normal(mean2, std2)

        return kl_divergence(distribution1, distribution2).float().to(self.device)  

class PolicyFunction():
    def __init__(self, gamma = 0.99, lam = 0.95, policy_kl_range = 0.03, policy_params = 2):
        self.gamma              = gamma
        self.lam                = lam
        self.policy_kl_range    = policy_kl_range 
        self.policy_params      = policy_params

    def monte_carlo_discounted(self, rewards, dones):
        running_add = 0
        returns     = []        
        
        for step in reversed(range(len(rewards))):
            running_add = rewards[step] + (1.0 - dones[step]) * self.gamma * running_add
            returns.insert(0, running_add)
            
        return torch.stack(returns)
      
    def temporal_difference(self, reward, next_value, done):
        q_values = reward + (1 - done) * self.gamma * next_value           
        return q_values
      
    def generalized_advantage_estimation(self, values, rewards, next_values, dones):
        gae     = 0
        adv     = []

        delta   = rewards + (1.0 - dones) * self.gamma * next_values - values
        for step in reversed(range(len(rewards))):
            gae = delta[step] + (1.0 - dones[step]) * self.gamma * self.lam * gae
            adv.insert(0, gae)
            
        return torch.stack(adv)

class Learner():
    def __init__(self, state_dim, action_dim, is_training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
                 minibatch, PPO_epochs, gamma, lam, learning_rate):        
        self.policy_kl_range    = policy_kl_range 
        self.policy_params      = policy_params
        self.value_clip         = value_clip    
        self.entropy_coef       = entropy_coef
        self.vf_loss_coef       = vf_loss_coef
        self.minibatch          = minibatch       
        self.PPO_epochs         = PPO_epochs
        self.is_training_mode   = is_training_mode
        self.action_dim         = action_dim 
        self.std                = torch.ones([1, action_dim]).float().to(device)       

        self.actor              = Actor_Model(state_dim, action_dim)
        self.actor_old          = Actor_Model(state_dim, action_dim)
        self.actor_optimizer    = Adam(self.actor.parameters(), lr = learning_rate)

        self.critic             = Critic_Model(state_dim, action_dim)
        self.critic_old         = Critic_Model(state_dim, action_dim)
        self.critic_optimizer   = Adam(self.critic.parameters(), lr = learning_rate)

        self.memory             = Memory()
        self.policy_function    = PolicyFunction(gamma, lam)  
        self.distributions      = Distributions()

        if is_training_mode:
          self.actor.train()
          self.critic.train()
        else:
          self.actor.eval()
          self.critic.eval()
    
    def save_all(self, states, actions, rewards, dones, next_states):
        self.memory.save_all(states, actions, rewards, dones, next_states)

    # Loss for PPO  
    def get_loss(self, action_mean, values, old_action_mean, old_values, next_values, actions, rewards, dones):
        # Don't use old value in backpropagation
        Old_values      = old_values.detach()

        # Finding the ratio (pi_theta / pi_theta__old):        
        logprobs        = self.distributions.logprob(action_mean, self.std, actions)
        Old_logprobs    = self.distributions.logprob(old_action_mean, self.std, actions).detach()       

        # Getting general advantages estimator
        Advantages      = self.policy_function.generalized_advantage_estimation(values, rewards, next_values, dones)
        Returns         = (Advantages + values).detach()
        Advantages      = ((Advantages - Advantages.mean()) / (Advantages.std() + 1e-6)).detach()        

        ratios          = (logprobs - Old_logprobs).exp()
        Kl              = self.distributions.kl_divergence(old_action_mean, self.std, action_mean, self.std)

        # Combining TR-PPO with Rollback (Truly PPO)
        pg_loss         = torch.where(
            (Kl >= self.policy_kl_range) & (ratios > 1),
            ratios * Advantages - self.policy_params * Kl,
            ratios * Advantages
        )        
        pg_loss         = pg_loss.mean()

        # Getting entropy from the action probability 
        dist_entropy    = self.distributions.entropy(action_mean, self.std).mean()

        # Getting critic loss by using Clipped critic value
        vpredclipped    = Old_values + torch.clamp(values - Old_values, -self.value_clip, self.value_clip) # Minimize the difference between old value and new value
        vf_losses1      = (Returns - values).pow(2) * 0.5 # Mean Squared Error
        vf_losses2      = (Returns - vpredclipped).pow(2) * 0.5 # Mean Squared Error        
        critic_loss     = torch.max(vf_losses1, vf_losses2).mean()                

        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss 
        loss            = (critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef) - pg_loss
        return loss

    # Get loss and Do backpropagation
    def training_ppo(self, states, actions, rewards, dones, next_states):                 
        action_mean, values          = self.actor(states), self.critic(states)
        old_action_mean, old_values  = self.actor_old(states), self.critic_old(states)
        next_values                  = self.critic(next_states)

        loss = self.get_loss(action_mean, values, old_action_mean, old_values, next_values, actions, rewards, dones)

        # === Do backpropagation ===
        
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        loss.backward()

        self.actor_optimizer.step() 
        self.critic_optimizer.step() 

        # === backpropagation has been finished ===

    # Update the model
    def update_ppo(self):        
        batch_size  = int(len(self.memory) / self.minibatch)
        dataloader  = DataLoader(self.memory, batch_size, shuffle = False)

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):       
            for states, actions, rewards, dones, next_states in dataloader:
                self.training_ppo(states.float().to(device), actions.float().to(device), rewards.float().to(device), \
                    dones.float().to(device), next_states.float().to(device))

        # Clear the memory
        self.memory.clear_memory()

        # Copy new weights into old policy:
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def get_weights(self):
        return self.actor.state_dict()

    def save_weights(self):
        torch.save(self.actor.state_dict(), 'agent.pth')

class Agent:  
    def __init__(self, state_dim, action_dim, is_training_mode):
        self.std                = torch.ones([1, action_dim]).float()
        self.is_training_mode   = is_training_mode   

        self.device             = torch.device('cpu')     

        self.memory             = Memory() 
        self.distributions      = Distributions(self.device)
        self.actor              = Actor_Model(state_dim, action_dim, torch.device('cpu'))        
        
        if is_training_mode:
          self.actor.train()
        else:
          self.actor.eval()

    def save_eps(self, state, action, reward, done, next_state):
        self.memory.save_eps(state, action, reward, done, next_state)
    
    def get_all(self):
        return self.memory.get_all()

    def act(self, state):
        state       = torch.FloatTensor(state).unsqueeze(0).to(self.device).detach()
        action_mean = self.actor(state)
        
        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            # Sample the action
            action  = self.distributions.sample(action_mean, self.std)
        else:
            action  = action_mean  
              
        return action.squeeze(0).numpy()
        
    def set_weights(self, weights):
        self.actor.load_state_dict(weights)

    def load_weights(self):
        self.actor.load_state_dict(torch.load('agent.pth', map_location = self.device))

@ray.remote
class Runner():
    def __init__(self, env_name, training_mode, render, n_update, tag):
        self.env                 = gym.make(env_name)
        self.states              = self.env.reset()

        self.state_dim           = self.env.observation_space.shape[0]
        self.action_dim          = self.env.action_space.shape[0]

        self.agent              = Agent(self.state_dim, self.action_dim, training_mode)

        self.render             = render
        self.tag                = tag
        self.training_mode      = training_mode
        self.n_update           = n_update

    def run_episode(self, i_episode, total_reward, eps_time):
        self.agent.load_weights()

        for _ in range(self.n_update):
            action                      = self.agent.act(self.states)
            next_state, reward, done, _ = self.env.step(action)
            
            eps_time        += 1 
            total_reward    += reward
            
            if self.training_mode:
                self.agent.save_eps(self.states.tolist(), action, reward, float(done), next_state.tolist())
                
            self.states = next_state 
                    
            if self.render:
                self.env.render()

            if done:
                self.states = self.env.reset()
                i_episode += 1
                print('Episode {} \t t_reward: {} \t time: {} \t process no: {} \t'.format(i_episode, total_reward, eps_time, self.tag))

                total_reward = 0
                eps_time = 0             
        
        return self.agent.get_all(), i_episode, total_reward, eps_time, self.tag

def plot(datas):
    print('----------')

    plt.plot(datas)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Datas')
    plt.show()

    print('Max :', np.max(datas))
    print('Min :', np.min(datas))
    print('Avg :', np.mean(datas))

def main():
    ############## Hyperparameters ##############
    training_mode       = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it

    render              = True # If you want to display the image, set this to True. Turn this off if you run this in Google Collab
    n_update            = 1024 # How many episode before you update the Policy. Recommended set to 1024 for Continous
    n_episode           = 100000 # How many episode you want to run
    n_agent             = 2 # How many agent you want to run asynchronously

    policy_kl_range     = 0.03 # Recommended set to 0.03 for Continous
    policy_params       = 5 # Recommended set to 5 for Continous
    value_clip          = 1.0 # How many value will be clipped. Recommended set to the highest or lowest possible reward
    entropy_coef        = 0.0 # How much randomness of action you will get. Because we use Standard Deviation for Continous, no need to use Entropy for randomness
    vf_loss_coef        = 1.0 # Just set to 1
    minibatch           = 32 # How many batch per update. size of batch = n_update / minibatch. Recommended set to 32 for Continous
    PPO_epochs          = 10 # How many epoch per update. Recommended set to 10 for Continous
    
    gamma               = 0.99 # Just set to 0.99
    lam                 = 0.95 # Just set to 0.95
    learning_rate       = 3e-4 # Just set to 0.95
    #############################################
    env_name            = 'BipedalWalker-v3'
    env                 = gym.make(env_name)
    state_dim           = env.observation_space.shape[0]
    action_dim          = env.action_space.shape[0]

    learner             = Learner(state_dim, action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
                            minibatch, PPO_epochs, gamma, lam, learning_rate)     
    #############################################
    start = time.time()
    ray.init()    
    try:
        runners = [Runner.remote(env_name, training_mode, render, n_update, i) for i in range(n_agent)]
        learner.save_weights()

        episode_ids = []
        for i, runner in enumerate(runners):
            episode_ids.append(runner.run_episode.remote(i, 0, 0))
            time.sleep(4)

        for _ in range(1, n_episode + 1):
            ready, not_ready = ray.wait(episode_ids)
            trajectory, i_episode, total_reward, eps_time, tag = ray.get(ready)[0]

            states, actions, rewards, dones, next_states = trajectory
            learner.save_all(states, actions, rewards, dones, next_states)

            learner.update_ppo()
            learner.save_weights()

            episode_ids = not_ready
            episode_ids.append(runners[tag].run_episode.remote(i_episode, total_reward, eps_time))
    except KeyboardInterrupt:        
        print('\nTraining has been Shutdown \n')
    finally:
        ray.shutdown()

        finish = time.time()
        timedelta = finish - start
        print('Timelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))

if __name__ == '__main__':
    main()