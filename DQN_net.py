import torch 
import numpy as np 
import random 
from collections import namedtuple
import torch.nn as nn 
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class MemoryBuffer(object):
    
    def __init__(self, capacity):
        self.cap = capacity 
        self.memory = []
    
    def add(self, s, a, s_prime, r):
        if len(self.memory) < self.cap:
            self.memory.append(Transition(state=s, action=a, next_state=s_prime, reward=r))
    
    def sample(self, batch_size):
        # returns a list of tranistions 
        return random.sample(self.memory, batch_size)
        
    def length(self):
        return len(self.memory)
    
    
class conv_net(nn.Module):

    def __init__(self, h, w, outputs):
        super(conv_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
    

    
