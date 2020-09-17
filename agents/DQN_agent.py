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
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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

class DQN_agent(object):
    
    def __init__(self, env, device, MB, q_net, q_target, n_episodes, T, 
                     gamma, e_start, e_end, e_decay, target_update, batch_size):

        self.n_episodes = n_episodes
        self.T = T 
        self.gamma = gamma 
        self.target_update = target_update 
        self.env = env 
        self.device = device 
        self.MB = MB
        self.q_net = q_net 
        self.q_target = q_target 
        self.e_start = e_start
        self.e_end = e_end 
        self.e_decay = e_decay 
        self.steps = 0
        self.n_actions = self.env.action_space.n
        self.batch_size = batch_size 
        self.optimizer = optim.RMSprop(self.q_net.parameters())
        self.losses = []
        
    def act(self, state):
        # use exponential decay to perform decayed-epsilon-greedy method 
        p = random.random()
        epsilon = self.e_start + (self.e_start-self.e_end)*math.exp(-self.steps/self.e_decay)
        self.steps+=1 
        
        if p > epsilon :
            with torch.no_grad():
                action = self.q_net(state).max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(self.n_actions)]], 
                                  device=self.device, dtype=torch.long)
        # shape: 1x1 to store in batches later 
        return action 
        
    def learn(self):
        
        if self.MB.length() < self.batch_size:
            return 
        
        # comes as a list of transitions with lenght batch size 
        transitions = self.MB.sample(self.batch_size)
        # makes a single Transition, 
        # where each element of namedtuple containes arrays of len batch size 
        batch = Transition(*zip(*transitions))
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        
        # q_net returns Q_values for all actions 
        # pass through q_net as a batch and obtain Q(s_t, a_t)
        q_s_a = self.q_net(states).gather(1, actions)
        
        # to update the estimate of Q(s_t, a_t) we must use target net that estimates V_s{t+1}
        # before passing there it is important to mask terminal states because:
        # Q(s_t, a_t) = r_t + gamma*V_max(s_{t+1}), 
        # if s_{t+1} is terminal we don't need the second term
        
        mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
        next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[mask] = self.q_target(next_states).max(1)[0].detach()
        
        est_q_s_a = rewards + next_state_values * self.gamma 

        # Compute Huber loss
        loss = F.smooth_l1_loss(q_s_a, est_q_s_a.unsqueeze(1))
        print("loss: ", loss)
        self.losses.append(loss.item())
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
    
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
            while done == False:
                print("t: ", t)
                action = self.act(state)
                # convert action tensor to python scalar and run env
                _, reward, done, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)

                last_screen = current_screen
                current_screen = get_screen(env, device)
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None
                self.MB.add(state, action, next_state, reward)
                state = next_state
                # update weights of q_net
                self.learn()
                t+=1
        # update target network 
        if i % self.target_update == 0:
            self.q_net.load_state_dict(self.q_target.state_dict())
        return self.losses
        