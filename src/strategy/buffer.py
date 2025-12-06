import numpy as np
import torch
import random

class Buffer:
    def __init__(self,
                 total_rollout_size,
                 device):
        self.capacity = total_rollout_size
        self.device = device

        self.states = []
        self.values = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.advantages = []
        self.returns = []
        self.dones = []

    def store_state(self, state):
        if len(self.states) < self.capacity:
            self.states.append(state)

    def store_value(self, value):
        if len(self.values) < self.capacity:
            self.values.append(value)

    def store_action(self, action):
        if len(self.actions) < self.capacity:
            self.actions.append(action)

    def store_log_prob(self, log_prob):
        if len(self.log_probs) < self.capacity:
            self.log_probs.append(log_prob)

    def store_rewards(self, reward):
        if len(self.rewards) < self.capacity:
            self.rewards.append(reward)

    def store_advantages(self, advantage):
        if len(self.advantages) < self.capacity:
            self.advantages.append(advantage)

    def store_returns(self, _return):
        if len(self.returns) < self.capacity:
            self.returns.append(_return)

    def store_dones(self, done):
        if len(self.dones) < self.capacity:
            self.dones.append(done)

    def get(self, req='state'):
        if req == 'state':
            return torch.tensor(np.array(self.states), dtype=torch.float32).to(self.device)
        if req == 'value':
            return torch.tensor(np.array(self.values), dtype=torch.float32).to(self.device)
        if req == 'action':
            return torch.tensor(np.array(self.actions), dtype=torch.float32).to(self.device)
        if req == 'log_prob':
            return torch.tensor(np.array(self.log_probs), dtype=torch.float32).to(self.device)
        if req == 'reward':
            return torch.tensor(np.array(self.rewards), dtype=torch.float32).to(self.device)
        if req == 'advantage':
            return torch.tensor(np.array(self.advantages), dtype=torch.float32).to(self.device)
        if req == 'return':
            return torch.tensor(np.array(self.returns), dtype=torch.float32).to(self.device)
        if req == 'done':
            return torch.tensor(np.array(self.dones), dtype=torch.float32).to(self.device)
        return None

    def clear(self):
        self.states = []
        self.values = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.advantages = []
        self.returns = []
        self.dones = []

    def random_shuffling(self):
        zipped_data = list(zip(self.states, self.values, self.actions, self.log_probs, self.rewards, self.advantages, self.returns, self.dones))
        random.shuffle(zipped_data)

        new_states, new_values, new_actions, new_log_probs, new_rewards, new_advantages, new_returns, new_dones = zip(*zipped_data)
        self.states = list(new_states)
        self.values = list(new_values)
        self.actions = list(new_actions)
        self.log_probs = list(new_log_probs)
        self.rewards = list(new_rewards)
        self.advantages = list(new_advantages)
        self.returns = list(new_returns)
        self.dones = list(new_dones)