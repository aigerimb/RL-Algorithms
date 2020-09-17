import torch 
import numpy as np 
import random 
from collections import namedtuple
import torch.nn as nn 
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import math 
import torch.optim as optim
import torch.distributions as distributions

def get_cart_location(screen_width, env):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen(env, device):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width, env)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

class A2C_agent(object):
    
    def __init__(self, n_episodes, env, device, actor, critic, lr_a, lr_c, gamma, test_interval):
        self.env = env 
        self.n_episodes = n_episodes 
        self.device = device 
        self.actor = actor
        self.critic = critic 
        self.gamma = gamma 
        self.actor_optim = optim.Adam(actor.parameters(), lr=lr_a)
        self.critic_optim = optim.Adam(critic.parameters(), lr=lr_c)
        self.test_interval = test_interval 
        
        
    def act(self, state):
        a =  self.actor(state)
        probs = F.softmax(self.actor(state))
        # Categorical distirbution takes p vector of probs
        # that specify the probability for each category/action
        m = distributions.Categorical(probs)
        action = m.sample()
        logprob = m.log_prob(action)
        
        return action, logprob 
  
    def learn(self, logprobs, values, rewards, normalize=True):
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma*R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, device=self.device)
        if normalize:
            returns = (returns - returns.mean()) / returns.std()
        
        A = returns-values
        if normalize:
            A =  (A - A.mean()) / A.std()
        actor_loss = -(A.detach() * logprobs).sum()
        critic_loss = F.smooth_l1_loss(returns, values).sum()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        return actor_loss.item(), critic_loss.item()
        

    def train(self):
        env = self.env 
        device = self.device 
        self.actor.train()
        self.critic.train()
        actor_losses = []
        critic_losses = []
        ep_rewards = []
        test_rewards = []
        for i in range(self.n_episodes):
            print("episode: ", i)
            env.reset()
            last_screen = get_screen(env, device)
            current_screen = get_screen(env, device)
            #state is defined as the difference between last and current screens 
            state = current_screen - last_screen
            done = False 
            t = 0
            logprobs = []
            rewards = []
            values = []
            ep_reward = 0
            while done == False:
                print("t: ", t)
                action, logprob = self.act(state)
                v = self.critic(state)
                # convert action tensor to python scalar and run env
                _, reward, done, _ = env.step(action.item())
                ep_reward += reward 
                logprobs.append(logprob)
                values.append(v)
                rewards.append(reward)
                last_screen = current_screen
                current_screen = get_screen(env, device)
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None
                state = next_state
                t+=1
                
          
            logprobs = torch.cat(logprobs)
            values = torch.cat(values).squeeze(-1)
            
            # optimize 
            actor_loss, critic_loss = self.learn(logprobs, values, rewards)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            ep_rewards.append(ep_reward)
            
            # perform tests 
            if  i%self.test_interval == 0:
                test_reward = self.test()
                test_rewards.append(test_reward)
                
        return actor_losses, critic_losses, ep_rewards, test_rewards
    def test(self):
        self.actor.eval()
        env = self.env 
        device = self.device 
        env.reset()
        last_screen = get_screen(env, device)
        current_screen = get_screen(env, device)
        state = current_screen - last_screen
        done = False 
        t= 0
        with torch.no_grad():
            ep_reward = 0
            while done == False:
                print("t: ", t)
                action, logprob = self.act(state)
                _, reward, done, _ = env.step(action.item())
                ep_reward += reward 
                last_screen = current_screen
                current_screen = get_screen(env, device)
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None
                state = next_state
                t+=1
        self.actor.train()
        return ep_reward