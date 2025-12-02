# import torch
# import numpy as np
#
#
# class Buffer:
#
#     def __init__(self, total_rollout_size, seq_len, input_dim, action_dim, device):
#         self.capacity = total_rollout_size + 1
#         self.device = device
#
#         self.states = torch.zeros((self.capacity, seq_len, input_dim), dtype=torch.float32)
#         self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.float32)
#         self.log_probs = torch.zeros(self.capacity, dtype=torch.float32)
#         self.values = torch.zeros(self.capacity, dtype=torch.float32)
#         self.rewards = torch.zeros(self.capacity, dtype=torch.float32)
#         self.dones = torch.zeros(self.capacity, dtype=torch.float32)
#         self.advantages = torch.zeros(self.capacity, dtype=torch.float32)
#         self.returns = torch.zeros(self.capacity, dtype=torch.float32)
#
#         self.current_step_state = 0
#         self.current_step_action = 0
#         self.current_step_log_prob = 0
#         self.current_step_value = 0
#         self.current_step_reward = 0
#         self.current_step_done = 0
#         self.current_step_advantage = 0
#         self.current_step_return = 0
#
#     def store_state(self, value):
#         if self.current_step_state < self.capacity and value is not None:
#             self.states[self.current_step_state] = torch.as_tensor(value, dtype=torch.float32)
#             self.current_step_state += 1
#         else:
#             print("Warning: Buffer is full. Not storing new data.")
#
#     def store_action(self, value):
#         if self.current_step_action < self.capacity and value is not None:
#             self.actions[self.current_step_action] = torch.as_tensor(value, dtype=torch.float32)
#             self.current_step_action += 1
#         else:
#             print("Warning: Buffer is full. Not storing new data.")
#
#     def store_log_prob(self, value):
#         if self.current_step_log_prob < self.capacity and value is not None:
#             self.log_probs[self.current_step_log_prob] = torch.as_tensor(value, dtype=torch.float32)
#             self.current_step_log_prob += 1
#         else:
#             print("Warning: Buffer is full. Not storing new data.")
#
#     def store_value(self, value):
#         if self.current_step_value < self.capacity and value is not None:
#             self.values[self.current_step_value] = torch.as_tensor(value, dtype=torch.float32)
#             self.current_step_value += 1
#         else:
#             print("Warning: Buffer is full. Not storing new data.")
#
#     def store_reward(self, value):
#         if self.current_step_reward < self.capacity and value is not None:
#             self.rewards[self.current_step_reward] = torch.as_tensor(value, dtype=torch.float32)
#             self.current_step_reward += 1
#         else:
#             print("Warning: Buffer is full. Not storing new data.")
#
#     def store_done(self, value):
#         if self.current_step_done < self.capacity and value is not None:
#             self.dones[self.current_step_done] = torch.as_tensor(value, dtype=torch.float32)
#             self.current_step_done += 1
#         else:
#             print("Warning: Buffer is full. Not storing new data.")
#
#     def store_advantage(self, value):
#         if self.current_step_advantage < self.capacity and value is not None:
#             self.advantages[self.current_step_advantage] = torch.as_tensor(value, dtype=torch.float32)
#             self.current_step_advantage += 1
#         else:
#             print("Warning: Buffer is full. Not storing new data.")
#
#     def store_return(self, value):
#         if self.current_step_return < self.capacity and value is not None:
#             self.returns[self.current_step_return] = torch.as_tensor(value, dtype=torch.float32)
#             self.current_step_return += 1
#         else:
#             print("Warning: Buffer is full. Not storing new data.")
#
#     def clear(self):
#         self.current_step = 0
#
#     def display(self):
#         """Helper to print shapes (for debugging)"""
#         print("--- Buffer Contents (all on CPU) ---")
#         print("states:", self.states.shape)
#         print("actions:", self.actions.shape)
#         print("log_probs:", self.log_probs.shape)
#         print("values:", self.values.shape)
#         print("rewards:", self.rewards.shape)
#         print("dones:", self.dones.shape)
#         print("advantages:", self.advantages.shape)
#         print("returns:", self.returns.shape)
#         print(f"current_step: {self.current_step_state} / {self.capacity}")

import torch
import numpy as np


class Buffer:
    def __init__(self, device='cpu'):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.advantages = []
        self.returns = []
        self.device = device

    def store(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def finish_path(self, last_val, gamma, gae_lambda):
        """
        Computes GAE (Generalized Advantage Estimation) and Returns
        at the end of a rollout.
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_val])  # Append last value for t+1 calc
        dones = np.array(self.dones)

        advantages = np.zeros_like(rewards)
        last_gae_lam = 0

        # Backward iteration for GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

        self.advantages = advantages.tolist()
        self.returns = (advantages + values[:-1]).tolist()

    def get_all(self):
        """
        Returns the entire buffer as tensors for the Full-Batch Update.
        """
        # Convert to tensors
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32).to(self.device)
        log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32).to(self.device)
        values = torch.tensor(np.array(self.values), dtype=torch.float32).to(self.device)
        advantages = torch.tensor(np.array(self.advantages), dtype=torch.float32).to(self.device)
        returns = torch.tensor(np.array(self.returns), dtype=torch.float32).to(self.device)

        return states, actions, log_probs, values, advantages, returns

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.advantages = []
        self.returns = []