import gym 
import numpy as np 
import matplotlib.pyplot as plt
import torch 
from A2C_agent import get_screen, A2C_agent
from a2c_net import conv_net, MLP

device = torch.device('cpu')
hidden_dim = 128
if __name__ == '__main__':
    # allows to access the envionment specific dynamics 
    env = gym.make('CartPole-v0').unwrapped
    env.reset()
    plt.figure()
    plt.imshow(get_screen(env, device).cpu().squeeze(0).permute(1, 2, 0).numpy(),
                       interpolation='none')
    plt.title('Example extracted screen')
    plt.show()
    init_screen = get_screen(env, device)
    actor = conv_net(init_screen.shape[2], init_screen.shape[3], env.action_space.n)
    critic = MLP(init_screen.shape[2], 1, hidden_dim)
    