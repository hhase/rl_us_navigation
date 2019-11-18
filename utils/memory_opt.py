# Replay Memory adapted from https://github.com/meagmohit/RL-playground/blob/master/Rainbow/memory.py

import torch
import numpy as np
from collections import namedtuple

Experience = namedtuple(
    # TODO - consider multi-frame state to include time related information
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False
        self.sum_tree = np.zeros((2 * size - 1), dtype=np.float32)      # Tree for storing probabilites
        self.data = np.array([None] * size)                             # Ring buffer
        self.max = 1

    def _propagate(self, index, value):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate(parent, value)

    def update(self, index, value):
        self.sum_tree[index] = value
        self._propagate(index, value)
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data                                    # Store data in ring-buffer
        self.update(self.index + self.size - 1, value)                  # Update leaf node corresponding to the ring-buffer
        self.index = (self.index + 1) % self.size
        self.full = self.full or self.index == 0
        self.max = max(value, self.max)

    def _retrieve(self, index, value):
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.sum_tree):
            return index
        elif value <= self.sum_tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.sum_tree[left])

    def find(self, value):
        index = self._retrieve(0, value)
        data_index = index - self.size + 1
        return (self.sum_tree[index], data_index, index)

    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]


class ReplayMemory():
    def __init__(self, args):
        self.device = args.device
        self.capacity = args.memory_size
        self.discount = args.gamma
        self.n = 0
        self.history = 1 #args.history_length         # Look into this when considering multiple frames for one experience
        self.beta = args.priority_weight
        self.priority_exponent = args.priority_exponent
        self.time_step = 0
        self.transitions = SegmentTree(self.capacity)

    def append(self, state, action, reward, next_state):
        state = state.mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))
        next_state = next_state.mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))
        self.transitions.append(Experience(state, action, next_state, reward), self.transitions.max)
        self.time_step += 1

    def _get_transition(self, index):
        transition = self.transitions.get(index)
        return transition

    def _get_sample_from_segment(self, segment, i):
        valid = False
        while not valid:
            sample = np.random.uniform(i * segment, (i + 1) * segment)
            prob, index, tree_index = self.transitions.find(sample)
            if (self.transitions.index - index) % self.capacity > self.n and (index - self.transitions.index) % self.capacity >= self.history and prob != 0:
                valid = True
        transition = self._get_transition(index)
        state = transition.state.to(device=self.device).to(dtype=torch.float32).div_(255)
        next_state = transition.next_state.to(device=self.device).to(dtype=torch.float32).div_(255)
        action = transition.action.clone().detach()
        reward = transition.reward.clone().detach()
        return prob, index, tree_index, state, action, reward, next_state

    def sample(self, batch_size):
        p_total = self.transitions.total()
        segment = p_total / batch_size
        batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]
        probs, idxs, tree_idxs, states, actions, rewards, next_states = zip(*batch)
        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)
        probs = np.array(probs, dtype=np.float32) / p_total
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.beta
        weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=self.device)
        return tree_idxs, states, actions, rewards, next_states, weights

    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]

    def update_beta(self, beta_step):
        self.beta += beta_step

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration
        state_stack = [None] * self.history
        state_stack[-1] = self.transitions.data[self.current_idx].state
        prev_timestep = self.transitions.data[self.current_idx].timestep
        for t in reversed(range(self.history - 1)):
            if prev_timestep == 0:
                state_stack[t] = None #blank_trans.state
            else:
                state_stack[t] = self.transitions.data[self.current_idx + t - self.history + 1].state
                prev_timestep -= 1
        state = torch.stack(state_stack, 0).to(dtype=torch.float32, device=self.device).div_(255)
        self.current_idx += 1
        return state
