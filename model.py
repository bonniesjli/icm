import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)

def swish(x):
    return x * F.sigmoid(x)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class MlpActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(MlpActorCriticNetwork, self).__init__()
        self.com_layer1 = nn.Linear(input_size, 256)
        self.batch_1 = nn.BatchNorm1d(256)
        self.com_layer2 = nn.Linear(256, 256)
        self.batch_2 = nn.BatchNorm1d(256)
        self.com_layer3 = nn.Linear(256, 256)
        self.batch_3 = nn.BatchNorm1d(256)

        self.actor_1 = nn.Linear(256, 256)
        self.actor_2 = nn.Linear(256, 256)
        self.actor = nn.Linear(256, output_size)
        self.critic_1 = nn.Linear(256, 256)
        self.critic_2 = nn.Linear(256, 256)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        
        x = swish(self.batch_1(self.com_layer1(x)))
        x = swish(self.batch_2(self.com_layer2(x)))
        x = swish(self.batch_3(self.com_layer3(x)))
        actor_1 = swish(self.actor_1(x))
        actor_2 = swish(self.actor_2(x))
        policy = self.actor(actor_2)
        critic_1 = swish(self.critic_1(x))
        critic_2 = swish(self.critic_2(x))
        value = self.critic(critic_2)

        return policy, value