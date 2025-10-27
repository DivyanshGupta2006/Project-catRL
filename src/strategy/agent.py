import torch
import torch.nn as nn
from numpy.ma.core import indices
from torch.optim import Adam

from src.strategy.model import Model

class Agent:
    def __init__(self,
                 input_dim,
                 lr,
                 gamma,
                 gae_lambda,
                 clip_epsilon,
                 value_loss_coef,
                 entropy_loss_coef,
                 device):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_loss_coef = entropy_loss_coef
        self.device = device

        self.model = Model(input_dim).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.value_loss_fn = nn.MSELoss()

    def _compute_gae(self, rewards, values, dones, last_value):
        advantages = torch.zeros_like(rewards).to(self.device)

        last_gae_lam = 0
        T = len(rewards)

        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t]
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_value = values[t+1]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam * next_non_terminal

        returns = advantages + values

        return advantages, returns

    def update(self, buffer, batch_size, update_epochs):
        states = buffer['states'].to(self.device)
        actions = buffer['actions'].to(self.device)
        old_log_probs = buffer['log_probs'].to(self.device)
        advantages = buffer['advantages'].to(self.device)
        returns = buffer['returns'].to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_samples = len(states)
        indices = torch.arange(n_samples)

        for _ in range(update_epochs):
            indices = indices[torch.randperm(n_samples)]

            for start in range(n_samples):
                end = start + batch_size
                if end > n_samples:
                    continue
                mb_indices = indices[start:end]
                new_actions, new_log_probs, entropy, new_values, _ = self.model.get_action_and_value(
                    states[mb_indices],
                    action=actions[mb_indices]
                )
                new_values = new_values.squeeze(-1)
                log_ratio = new_log_probs - old_log_probs[mb_indices]
                ratio = torch.exp(log_ratio)
                surr1 = ratio * advantages[mb_indices]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages[mb_indices]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.value_loss_fn(new_values, returns[mb_indices])
                entropy_loss = -entropy.mean()
                total_loss = (
                        policy_loss +
                        self.value_loss_coef * value_loss +
                        self.entropy_loss_coef * entropy_loss
                )
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # Gradient clipping
                self.optimizer.step()
