import gym 
import numpy as np 
import matplotlib.pyplot as plt
import torch 
from agents.A2C_agent import get_screen, A2C_agent
from neuralnets.a2c_net import conv_net

device = torch.device('cpu')
lr_a = 0.001
lr_c = 0.0001
n_episodes = 81
gamma = 0.99 
test_interval = 10
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
    critic = conv_net(init_screen.shape[2], init_screen.shape[3], 1)
    agent = A2C_agent(n_episodes, env, device, actor, critic, lr_a, lr_c, gamma, test_interval)
    actor_loss, critic_loss, train_rewards, test_rewards = agent.train()
    plt.plot(np.arange(len(train_rewards)), train_rewards, color ='b', label='train rewards')
    plt.plot(np.arange(0, n_episodes, test_interval), test_rewards, color ='r', label='test rewards')
    plt.legend()
    plt.title('A2C Training')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig('A2CTrain.png')
    
