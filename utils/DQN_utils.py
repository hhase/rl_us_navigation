# Noisy Layer implementation taken from https://github.com/Kaixhin/Rainbow

import math
import torch
import random
import numpy as np
import torch.nn as nn

def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size + kernel_size + 2 * padding) / stride) + 1
    return output

class DQN(nn.Module):
    def __init__(self, args, im_dims, num_actions):
        super(DQN, self).__init__()
        im_height, im_width = im_dims[-2], im_dims[-1]
        self.noisy = args.noisy_network
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(int(im_height/4) * int(im_width/4) * 64, 1000, bias=False)
        self.fc2 = nn.Linear(1000, num_actions, bias=False)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class Dueling_DQN(nn.Module):
    def __init__(self, args, im_dims, num_actions):
        super(Dueling_DQN, self).__init__()
        self.get_state_value = False
        im_height, im_width = im_dims[-2], im_dims[-1]
        self.noisy = args.noisy_network
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        if self.noisy:
            self.advantage = nn.Sequential(
                NoisyLinear(int(im_height / 32) * int(im_width / 32) * 512, 1000),
                nn.ReLU(),
                NoisyLinear(1000, num_actions, bias=False, std_init=1.0)
            )
            self.value = nn.Sequential(
                NoisyLinear(int(im_height / 32) * int(im_width / 32) * 512, 1000),
                nn.ReLU(),
                NoisyLinear(1000, 1, bias=False, std_init=1.0)
            )
            print("Noisy Layers set up successfully!")
        else:
            self.advantage = nn.Sequential(
                nn.Linear(int(im_height / 32) * int(im_width / 32) * 512, 1000, bias=False),
                nn.ReLU(),
                nn.Linear(1000, num_actions, bias=False)
            )
            self.value = nn.Sequential(
                nn.Linear(int(im_height / 32) * int(im_width / 32) * 512, 1000, bias=False),
                nn.ReLU(),
                nn.Linear(1000, 1, bias=False)
            )

    def forward(self, x, state_value=False):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        advantage = self.advantage(out)
        value = self.value(out)
        if state_value:
            return value
        else:
            return value + advantage - advantage.mean()

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias = bias
        if self.bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
            self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        if self.bias:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        if self.bias:
            self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            if self.bias:
                return nn.functional.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
            else:
                return nn.functional.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon)
        else:
            if self.bias:
                return nn.functional.linear(input, self.weight_mu, self.bias_mu)
            else:
                return nn.functional.linear(input, self.weight_mu)


class EpsilonGreedyStrategy():
    def __init__(self, args):
        self.start = args.eps_start
        self.end = args.eps_end
        self.decay = args.eps_decay
        self.begin = args.eps_decay_begin

    def get_exploration_rate(self, current_step):
        return 1.0 if current_step < self.begin else self.end + (self.start - self.end) * math.exp(-1 * (current_step - self.begin) * self.decay)


class GreedyStrategy():
    def __init__(self, args):
        pass

    def get_exploration_rate(self, timestep):
        return 0


class Agent():
    def __init__(self, strategy, num_actions, device):
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net, timestep):
        rate = self.strategy.get_exploration_rate(timestep)

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)  # explore
        else:
            with torch.no_grad():
                action = policy_net(state).to(self.device)
                return action.argmax(dim=1)  # exploit

class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current_q_values(policy_net, states, actions):
        policy_net.eval()
        q_values = policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
        policy_net.train()
        return q_values

    @staticmethod
    def get_next_actions(policy_net, next_states):
        policy_net.eval()
        next_actions = policy_net(next_states).argmax(dim=1).detach().cpu()
        policy_net.train()
        return next_actions

    @staticmethod
    def get_next_q_values(target_net, next_states, next_actions, nops):
        final_state_locations = nops
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        next_actions = next_actions[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
       #values[non_final_state_locations] = target_net(non_final_states).gather(dim=1, index=next_actions.unsqueeze(-1))
       #values[non_final_state_locations] = target_net(non_final_states)[[list(range(len(next_actions))), next_actions]].detach()
        return values


def get_TD_errors(policy_net, target_net, states, actions, rewards, next_states, gamma):
    policy_net.eval()
    td_errors = []
    for i in range(len(actions)):
        policy_net_output       = policy_net(states[i].unsqueeze(0)).detach().cpu().numpy().squeeze()
        policy_net_output_next  = policy_net(next_states[i].unsqueeze(0)).detach().cpu().numpy().squeeze()
        target_net_output       = target_net(next_states[i].unsqueeze(0)).detach().cpu().numpy().squeeze()
        action                  = actions[i].detach().cpu().numpy()
        reward                  = rewards[i].detach().cpu().numpy()
        td_error = reward + gamma * target_net_output[np.argmax(policy_net_output_next)] - policy_net_output[action]
        td_error = td_error.squeeze()
        td_errors.append(np.abs(td_error))
    policy_net.train()
    return td_errors

