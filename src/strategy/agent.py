import torch
import torch.nn as nn
from torch.optim import Adam

class Agent:
    def __init__(self,
                 model,
                 gamma,
                 gae_lambda,
                 clip_epsilon,
                 value_loss_coef,
                 entropy_loss_coef,
                 learning_rate,
                 device,
                 model_path):

        self.device = device
        self.model = model.to(self.device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_loss_coef = entropy_loss_coef
        self.learning_rate = learning_rate
        self.model_path = model_path

        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.value_loss_fn = nn.MSELoss()

    def get_action_and_value(self, buffer):
        x = buffer.states[-1]
        dist, value = self.model.forward(x)
        buffer.values = value.tolist()
        action = dist.sample()
        buffer.actions = action.tolist()
        log_prob = dist.log_prob(action).sum(dim=-1)
        buffer.log_probs = log_prob
        entropy = dist.entropy().sum(dim=-1)
        buffer.entropies = entropy.tolist()
        return buffer

    def _compute_gae(self, buffer, next_value, next_done):
        rewards = torch.tensor(buffer.rewards, dtype=torch.float32).to(self.device)
        values = torch.tensor(buffer.values, dtype=torch.float32).to(self.device)
        dones = torch.tensor(buffer.dones, dtype=torch.float32).to(self.device)

        T = len(rewards)
        advantages = torch.zeros_like(rewards).to(self.device)

        last_gae_lam = 0

        all_values = torch.cat([values, next_value.unsqueeze(0)], dim=0)
        all_dones = torch.cat([dones, next_done.unsqueeze(0)], dim=0)

        for t in reversed(range(T)):
            value_t = all_values[t]
            value_t_plus_1 = all_values[t + 1]
            next_non_terminal = 1.0 - all_dones[t + 1]

            delta = rewards[t] + self.gamma * value_t_plus_1 * next_non_terminal - value_t

            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam * next_non_terminal

        returns = advantages + values

        buffer.advantages = advantages.cpu().numpy().tolist()
        buffer.returns = returns.cpu().numpy().tolist()


    def update(self, buffer):
        # Get all data from the buffer as flat tensors
        obs_tensor, actions_tensor, old_log_probs_tensor, \
            advantages_tensor, returns_tensor = buffer.get_all_tensors(self.device)

        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Log running losses
        total_loss_log, policy_loss_log, value_loss_log, entropy_log = 0, 0, 0, 0
        batches_processed = 0

        # Run PPO epochs
        for _ in range(self.epochs):

            # Get a generator for sampling mini-batches of sequences
            sampler = buffer.sample(self.batch_size, self.sequence_length)

            for obs_batch, actions_batch, old_log_probs_batch, \
                    advantages_batch, returns_batch in sampler:
                # --- CRITICAL ---
                # This assumes your model.py's forward pass is fixed
                # to return the output for *all* timesteps in the sequence.

                # obs_batch shape: [128, 72, 126]

                # Get zeroed initial hidden state for the batch
                h0, c0 = self.model.get_initial_states(self.batch_size, self.device)

                # Forward pass: Re-evaluate the states in the batch
                dist, values, _ = self.model(obs_batch, (h0, c0))

                # values shape: [128, 72, 1] -> [128, 72]
                values = values.squeeze(-1)

                # Get new log_probs and entropy
                # new_log_probs shape: [128, 72]
                new_log_probs = dist.log_prob(actions_batch).sum(dim=-1)

                # entropy shape: [128, 72] -> scalar
                entropy = dist.entropy().sum(dim=-1).mean()

                # --- PPO Loss Calculation ---

                # 1. Policy Loss (Clip)
                # ratio shape: [128, 72]
                ratio = torch.exp(new_log_probs - old_log_probs_batch)

                # surr1/surr2 shape: [128, 72]
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_batch

                policy_loss = -torch.min(surr1, surr2).mean()

                # 2. Value Loss (MSE)
                value_loss = self.value_loss_fn(values, returns_batch)

                # 3. Total Loss
                total_loss = (
                        policy_loss +
                        self.value_loss_coef * value_loss -
                        self.entropy_loss_coef * entropy
                )

                # --- Gradient Step ---
                self.optimizer.zero_grad()
                total_loss.backward()
                # Clip gradients to prevent exploding gradients
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                # --- Logging ---
                total_loss_log += total_loss.item()
                policy_loss_log += policy_loss.item()
                value_loss_log += value_loss.item()
                entropy_log += entropy.item()
                batches_processed += 1

        # Return mean losses
        return {
            "total_loss": total_loss_log / batches_processed,
            "policy_loss": policy_loss_log / batches_processed,
            "value_loss": value_loss_log / batches_processed,
            "entropy": entropy_log / batches_processed
        }

    def save(self):
        print("Saving the model...")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.model_path)

    def load(self):
        print("Loading the model...")
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.to(self.device)
        except Exception as e:
            print(f"--- Error loading models: {e} ---")