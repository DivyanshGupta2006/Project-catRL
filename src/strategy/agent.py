# import torch
# import torch.nn as nn
# from torch.optim import Adam
# from src.utils import get_config
#
# config = get_config.read_yaml()
#
# class Agent:
#     def __init__(self,
#                  model,
#                  gamma,
#                  gae_lambda,
#                  clip_epsilon,
#                  value_loss_coef,
#                  entropy_loss_coef,
#                  learning_rate,
#                  mini_batch_size,
#                  device,
#                  model_path):
#
#         self.device = device
#         self.model = model.to(self.device)
#
#         self.gamma = gamma
#         self.gae_lambda = gae_lambda
#         self.clip_epsilon = clip_epsilon
#         self.value_loss_coef = value_loss_coef
#         self.entropy_loss_coef = entropy_loss_coef
#         self.learning_rate = learning_rate
#         self.model_path = model_path
#         self.mini_batch_size = mini_batch_size
#
#         self.optimizer = Adam(model.parameters(), lr=learning_rate)
#         self.value_loss_fn = nn.MSELoss()
#
#     def get_action_and_value(self, buffer):
#         x = buffer.states[buffer.current_step_state - 1].unsqueeze(0).to(self.device)
#         dist, value = self.model.forward(x)
#         buffer.store_value(value)
#         action = dist.sample()
#         buffer.store_action(action)
#         log_prob = dist.log_prob(action).sum(dim=-1)
#         buffer.log_probs = log_prob
#         return buffer
#
#     def get_new_model_output(self, states, actions):
#         dist, new_values = self.model.forward(states)
#         new_log_prob = dist.log_prob(actions).sum(dim=-1)
#         entropies = dist.entropy().sum(dim=-1)
#         return new_log_prob, new_values, entropies
#
#     def compute_gae(self, buffer, next_value, next_done):
#         rewards = buffer.rewards.to(self.device)
#         values = buffer.values.to(self.device)
#         dones = buffer.dones.to(self.device)
#
#         T = len(rewards)
#         advantages = torch.zeros_like(rewards).to(self.device)
#         print(advantages)
#
#         last_gae_lam = 0
#
#         all_values = torch.cat([values, next_value.to(self.device)], dim=0)
#         all_dones = torch.cat([dones, next_done.to(self.device)], dim=0)
#
#         for t in reversed(range(T)):
#             value_t = all_values[t]
#             value_t_plus_1 = all_values[t + 1]
#             next_non_terminal = 1.0 - all_dones[t + 1]
#
#             delta = rewards[t] + self.gamma * value_t_plus_1 * next_non_terminal - value_t
#
#             advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam * next_non_terminal
#
#         returns = advantages + values
#
#         buffer.advantages = advantages.cpu().detach()
#         buffer.returns = returns.cpu().detach()
#
#         return buffer
#
#
#     def update(self, buffer, start):
#         states = torch.tensor(buffer.states[start:start+self.mini_batch_size], dtype=torch.float32).to(self.device)
#         actions = torch.tensor(buffer.actions[start:start+self.mini_batch_size], dtype=torch.float32).to(self.device)
#         old_log_probs = torch.tensor(buffer.log_probs[start:start+self.mini_batch_size], dtype=torch.float32).to(self.device)
#         advantages = torch.tensor(buffer.advantages[start:start+self.mini_batch_size], dtype=torch.float32).to(self.device)
#         returns = torch.tensor(buffer.returns[start:start+self.mini_batch_size], dtype=torch.float32).to(self.device)
#
#         num_samples = len(states)
#         if num_samples == 0:
#             return
#
#         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
#
#         new_log_probs, new_values, entropies = self.get_new_model_output(states,actions)
#
#         value_loss = self.value_loss_fn(new_values, returns).mean()
#         ratio = torch.exp(new_log_probs - old_log_probs)
#         surr1 = ratio * advantages
#         surr2 = torch.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
#         policy_loss = torch.min(surr1,surr2).mean()
#         entropy_loss = entropies.mean()
#
#         total_loss = self.value_loss_coef*value_loss - policy_loss - self.entropy_loss_coef * entropy_loss
#
#         # Backpropagate the loss
#         self.optimizer.zero_grad()
#         total_loss.backward()
#         self.optimizer.step()
#
#
#     def save(self):
#         print("Saving the model...")
#         torch.save({
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#         }, self.model_path)
#
#     def load(self):
#         print("Loading the model...")
#         try:
#             checkpoint = torch.load(self.model_path, map_location=self.device)
#             self.model.load_state_dict(checkpoint['model_state_dict'])
#             self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#             self.model.to(self.device)
#         except Exception as e:
#             print(f"--- Error loading models: {e} ---")

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np


class Agent:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.device = device

        # Hyperparameters
        self.gamma = config['hyperparameters']['gamma']
        self.gae_lambda = config['hyperparameters']['gae_lambda']
        self.clip_epsilon = config['hyperparameters']['clip_epsilon']
        self.value_loss_coef = config['hyperparameters']['value_loss_coef']
        self.entropy_loss_coef = config['hyperparameters']['entropy_loss_coef']
        self.learning_rate = config['hyperparameters']['learning_rate']
        self.epochs = config['hyperparameters']['num_epochs']

        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.mse_loss = nn.MSELoss()

    def get_action(self, state):
        """
        Forward pass for the rollout phase.
        Input State: (Seq_Len, Features) -> Converted to (1, Seq_Len, Features)
        """
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            # Hidden state is None -> initialized to 0 inside model for every window
            dist, value = self.model.forward(state_tensor)

            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action.cpu().numpy()[0], log_prob.cpu().item(), value.cpu().item()

    def get_value(self, state):
        """Helper to get value of state"""
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            _, value = self.model.forward(state_tensor)
        return value.cpu().item()

    def update(self, buffer):
        """
        PPO Update Phase (Algorithm II: M=T).
        Updates on the entire buffer K times.
        """
        self.model.train()

        # 1. Get Full Batch
        states, actions, old_log_probs, values, advantages, returns = buffer.get_all()

        # Normalize Advantages (Critical for stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 2. Epoch Loop
        for _ in range(self.epochs):
            # Forward pass on ENTIRE batch at once
            # States shape: (Batch=T, Seq_Len, Features)
            dist, new_values = self.model.forward(states)
            new_values = new_values.squeeze(-1)  # Match shape with returns

            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            # 3. Calculate Losses

            # Ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Surrogate Loss (Actor)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value Loss (Critic)
            critic_loss = self.mse_loss(new_values, returns)

            # Total Loss
            loss = actor_loss + (self.value_loss_coef * critic_loss) - (self.entropy_loss_coef * entropy)

            # 4. Backprop
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)  # Gradient Clipping
            self.optimizer.step()

        return loss.item()  # Return last loss for logging

    def save(self, path):
        torch.save(self.model.state_dict(), path)