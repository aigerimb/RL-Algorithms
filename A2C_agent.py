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
    
    def __init__(self, n_episodes, env, device, actor, critic, lr_a, lr_c):
        self.env = env 
        self.n_episodes = n_episodes 
        self.device = device 
        self.actor = actor
        self.critic = critic 
        self.lr_a = lr_a
        self.lr_c = lr_c 
        
        
    def act(self, state):
        
        probs = F.softmax(self.actor(state))
        # Categorical distirbution takes p vector of probs
        # that specify the probability for each category/action
        m = distributions.Categorical(probs)
        action = m.sample()
        logprob = m.log_prob(action)
        
        return action, logprob 
  
   # def learn(self):
        

    def train(self):
        env = self.env 
        device = self.device 
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
            actions = []
            rewards = []
            while done == False:
                print("t: ", t)
                action, logprob = self.act(state)
                v = self.critic(state)
                # convert action tensor to python scalar and run env
                _, reward, done, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)
                actions.append(action)
                logprobs.append([logprob])
                values.append([v])
                rewards.append(reward)
                last_screen = current_screen
                current_screen = get_screen(env, device)
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None
                self.MB.add(state, action, next_state, reward)
                state = next_state
                t+=1
                
        
   # def test(self):