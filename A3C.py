import torch 
import numpy as np 
import random 
from collections import namedtuple
import torch.nn as nn 
import torch.nn.functional as F
import gym 
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.multiprocessing as mp
import time
import os 
import torch.distributions as distributions
torch.autograd.set_detect_anomaly(True)


class MLP(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim):
        super(MLP, self).__init__()
        self.ac1 = nn.Linear(input_size, hidden_dim)
        self.ac2 = nn.Linear(hidden_dim, output_size)
        self.cr1 = nn.Linear(input_size, hidden_dim)
        self.cr2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        y = F.relu(self.ac1(x))
        output = self.ac2(y)
        v = F.relu(self.cr1(x))
        v = self.cr2(v)
        
        return output, v 
    
    def act(self, state):
        a, v =  self.forward(torch.from_numpy(state.astype(np.float32)))
        probs = F.softmax(a)
        # Categorical distirbution takes p vector of probs
        # that specify the probability for each category/action
        m = distributions.Categorical(probs)
        action = m.sample()
        logprob = m.log_prob(action)
        
        return action, logprob, v
    
    
def train(global_model, rank):
    
    torch.manual_seed(rank)
    local_model = MLP(input_size, output_size, hidden_dim)
    
    local_model.train()
    local_model.load_state_dict(global_model.state_dict())
    optimizer = optim.Adam(global_model.parameters(), lr=lr_a)

    optimizer.zero_grad()
    actor_losses = []
    critic_losses = []
    ep_rewards = []
    test_rewards = []
    for i in range(n_episodes):
        print("episode: ", i)
        env = gym.make('CartPole-v0').unwrapped
        state = env.reset()
        
        done = False 
        t = 0
        logprobs = []
        rewards = []
        values = []
        ep_reward = 0
        while done == False:
            print("t: ", t)
            action, logprob, v = local_model.act(state)
            # convert action tensor to python scalar and run env
            state, reward, done, _ = env.step(action.item())
            ep_reward += reward 
            logprobs.append(logprob)
            values.append(v)
            rewards.append(reward)
            t+=1
                
    #    print(logprobs)
    #    logprobs = torch.cat(logprobs)
    #    values = torch.cat(values).squeeze(-1)
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma*R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=device)
        for j in range(t):
            A = returns[j] - values[j]
            loss =  A.detach()*logprobs[j] + A**2
            loss.backward()
            
        for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
            global_param._grad = local_param.grad
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict())
            optimizer.zero_grad()
            
            
            
def test(global_model):
    env = gym.make('CartPole-v0').unwrapped
    state = env.reset()
    ep_reward = 0
    done = False 
    while done == False:
        
        action, logprob, v = global_model.act(state)
        # convert action tensor to python scalar and run env
        state, reward, done, _ = env.step(action.item())
        ep_reward += reward 
    
    test_rewards.append(ep_reward)
    
    
    
    
    
    
device = torch.device('cpu')
env = gym.make('CartPole-v0').unwrapped
lr_a = 0.001
n_episodes = 10
gamma = 0.99 
test_interval = 10    
n_train_processes = 8
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
hidden_dim = 128
test_rewards = []


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = " "
    global_model = MLP(input_size, output_size, hidden_dim)
    global_model.share_memory()

    processes = []
    s_t = time.time()
    for rank in range(n_train_processes + 1):  # + 1 for test process
        if rank == 0:
            p = mp.Process(target=test, args=(global_model, ))
        else:
            p = mp.Process(target=train, args=(global_model, rank,))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()
    e_t = time.time() - s_t
    path = 'A3C_params.pkl'
    torch.save(global_model.state_dict(), path)
    print("end time: ", e_t)
    
#global_model = MLP(input_size, output_size, hidden_dim)
#global_model.share_memory()    
#train(global_model, 1)