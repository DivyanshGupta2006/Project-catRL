import numpy as np
import torch
import torch.nn as nn

class Agent:

    def __init__(self,
                 model,
                 gamma,
                 gae_lambda,
                 clip_epsilon,
                 num_epochs,
                 mini_batch_size,
                 learning_rate,
                 value_loss_coef,
                 entropy_loss_coef,
                 model_path,
                 device):
        self.model = model
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.value_loss_coef = value_loss_coef
        self.entropy_loss_coef = entropy_loss_coef
        self.model_path = model_path
        self.device = device

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss()

    def get_action_and_value(self, buffer):
        x = buffer.get('state')[-1]

        dist, value = self.model.forward(x)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        buffer.store_action(action[0])
        buffer.store_value(value[0].item())
        buffer.store_log_prob(log_prob[0].item())

        return buffer

    def _compute_gae(self, buffer, next_value):
        rewards = buffer.get('rewards')
        values = buffer.get('values')

        advantages = np.zeros_like(rewards)
        last_gae_lambda = 0.0

        print(rewards)

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_value
            else:
                next_value = values[t+1]

            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = last_gae_lambda = delta + self.gamma * self.gae_lambda * last_gae_lambda

        buffer.advantages = advantages.tolist()
        buffer.returns = (advantages + values).tolist()

        return buffer

    def update(self, buffer, next_value):
        self.model.train()

        buffer = self._compute_gae(buffer, next_value)

        states = buffer.get('state')
        actions = buffer.get('action')
        log_probs = buffer.get('log_prob')
        rewards = buffer.get('reward')
        advantages = buffer.get('advantage')
        returns = buffer.get('return')

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        num_mini_batches = int(len(rewards) / self.mini_batch_size)

        for _ in range(self.num_epochs):
            for mini_batch in range(num_mini_batches):
                cur_states = states[mini_batch*self.mini_batch_size:(mini_batch+1)*self.mini_batch_size]
                cur_log_probs = log_probs[mini_batch*self.mini_batch_size:(mini_batch+1)*self.mini_batch_size]
                cur_advantages = advantages[mini_batch*self.mini_batch_size:(mini_batch+1)*self.mini_batch_size]
                cur_returns = returns[mini_batch*self.mini_batch_size:(mini_batch+1)*self.mini_batch_size]

                dist, new_values = self.model.forward(cur_states)
                new_values = new_values.squeeze(-1)

                new_log_probs = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_log_probs - cur_log_probs)

                surrogate1 = ratio * cur_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

                actor_loss = -torch.min(surrogate1, surrogate2).mean()

                critic_loss = self.loss(new_values, cur_returns)

                loss = actor_loss + critic_loss * self.value_loss_coef + self.entropy_loss_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

    def save(self):
        print("Saving the model...")
        torch.save(self.model.state_dict(), self.model_path)

    def load(self):
        print("Loading the model...")
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
        except Exception as e:
            print(f"--- Error loading models: {e} ---")
