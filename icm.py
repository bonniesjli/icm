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
    
class ICMModel(nn.Module):
    """ICM model for non-vision based tasks"""
    def __init__(self, input_size, output_size):
        super(ICMModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.resnet_time = 4
        self.device = device

        self.feature = nn.Sequential(
            nn.Linear(self.input_size, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 256),
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.BatchNorm1d(256),
            Swish(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            Swish(),
            nn.Linear(256, self.output_size)
        )

        self.residual = [nn.Sequential(
            nn.Linear(self.output_size + 256, 256),
            Swish(),
            nn.Linear(256, 256),
        ).to(self.device)] * 2 * self.resnet_time

        self.forward_net_1 = nn.Sequential(
            nn.Linear(self.output_size + 256, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 256)
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(self.output_size + 256, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 256)
        )

    def forward(self, state, next_state, action):

        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)
        # ---------------------

        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

        # residual
        for i in range(self.resnet_time):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig

        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))

        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action
    
class ICM():
    """Intrinsic Curisity Module"""
    def __init__(
        self, 
        state_size,
        action_size,
        learning_rate = 1e-4,
        eta = 0.01):
        
        self.model = ICMModel(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)
        
        self.input_size = state_size
        self.output_size = action_size
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.eta = eta
        self.device = device
    
    def compute_intrinsic_reward(self, state, next_state, action):
        """
        Compute intrinsic rewards for parallel transitions
        :param: (ndarray) states eg. [[state], [state], ... , [state]]
        :param: (ndarray) actions eg. [[action], [action], ... , [action]]
        :param: (ndarray) next_states eg. [[state], [state], ... , [state]]
        """
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)

        action_onehot = torch.FloatTensor(len(action), self.output_size).to(self.device)
        action_onehot.zero_()
        action_onehot.scatter_(1, action.view(len(action), -1), 1)

        real_next_state_feature, pred_next_state_feature, pred_action = self.model(
            state, next_state, action_onehot)
        intrinsic_reward = self.eta * (real_next_state_feature - pred_next_state_feature).pow(2).sum(1) / 2

        return intrinsic_reward.data.cpu().numpy()
    
    def train(self, states, next_states, actions):
        """
        Train the ICM model
        :param: (float tensors) states eg. [[state], [state], ... , [state]]
        :param: (long tensors) actions eg. [action, action, ... , action]
        :param: (float tensors) next_states eg. [[state], [state], ... , [state]]
        """
        action_onehot = torch.FloatTensor(len(actions), self.output_size).to(self.device)
        action_onehot.zero_()
        action_onehot.scatter_(1, actions.view(-1, 1), 1)
        real_next_state_feature, pred_next_state_feature, pred_action = self.model(
            states, next_states, action_onehot)
        
        inverse_loss = self.ce(
            pred_action, actions)

        forward_loss = self.mse(
            pred_next_state_feature, real_next_state_feature.detach())
        
        """compute joint loss??? """
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        
        loss = inverse_loss + forward_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()