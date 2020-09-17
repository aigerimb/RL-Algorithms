import gym 
import numpy as np 
import matplotlib.pyplot as plt
import torch 
from agents.DQN_agent import DQN_agent, get_screen
from neuralnets.DQN_net import MemoryBuffer, conv_net

device = torch.device('cpu')
mem_cap = 100000 
n_episodes = 10
T = 15
gamma = 0.99
epsilon =0.86
e_start = 0.9
e_end =0.05
e_decay = 200
target_update = 4
batch_size = 32

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
    MB = MemoryBuffer(mem_cap)
    q_net = conv_net(init_screen.shape[2], init_screen.shape[3], env.action_space.n)
    q_target = conv_net(init_screen.shape[2], init_screen.shape[3], env.action_space.n)
    q_target.load_state_dict(q_net.state_dict())
    q_target.eval()
    agent = DQN_agent(env, device, MB, q_net, q_target, n_episodes, T, 
                          gamma, e_start, e_end, e_decay, target_update, batch_size)
    plt.figure()
    losses = agent.train()
    plt.plot(np.arange(len(losses)), losses)
    plt.ylabel('loss')
    plt.xlabel('episodes')
    plt.title("Training")
    plt.savefig('DQN_training.png')
    


