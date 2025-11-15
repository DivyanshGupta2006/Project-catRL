import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal

# Import your model (assuming it's in 'src.strategy.model')
from src.strategy.model import Model
# We will use the Buffer from our previous discussion, NOT the one in your repo
from src.strategy.buffer import Buffer


class Agent:
    """
    The PPO Agent class, which manages data collection (rollout)
    and model updates (learning).
    """

    def __init__(self,
                 model,
                 device,
                 learning_rate=3e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_epsilon=0.2,
                 epochs=10,
                 batch_size=128,
                 sequence_length=72,
                 value_loss_coef=0.5,
                 entropy_loss_coef=0.01):
        """
        Initializes the agent.

        Args:
            model (nn.Module): The ActorCriticLSTM model.
            device (torch.device): CPU or CUDA.
            All other args are PPO hyperparameters.
        """

        self.model = model
        self.device = device

        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.value_loss_coef = value_loss_coef
        self.entropy_loss_coef = entropy_loss_coef

        # Optimizer
        self.optimizer = Adam(model.parameters(), lr=learning_rate)

        # Loss function
        self.value_loss_fn = nn.MSELoss()

    def get_action_and_value(self, obs_tensor, hidden_state):
        """
        Helper function to get action, log_prob, and value from the model.
        Used during rollouts (one step at a time).

        Args:
            obs_tensor (torch.Tensor): Shape [1, 1, input_dim]
            hidden_state (tuple): (h_0, c_0)

        Returns:
            Tuple of (action, log_prob, value, next_hidden_state)
        """

        # --- CRITICAL ---
        # This assumes your model.py's forward pass is fixed
        # to process all timesteps, not just the last one.

        # Get distribution and value from model
        # dist is a Normal distribution
        # value shape: [1, 1, 1]
        dist, value, next_hidden_state = self.model(obs_tensor, hidden_state)

        # Sample an action from the distribution
        action = dist.sample()  # Shape: [1, 1, 10]

        # Calculate the log-probability of that action
        # We must sum the log_probs of the 10 independent actions
        log_prob = dist.log_prob(action).sum(dim=-1)  # Shape: [1, 1]

        # Squeeze outputs to scalars/vectors for storage
        action = action.squeeze(0).squeeze(0)  # Shape: [10]
        log_prob = log_prob.squeeze(0).squeeze(0)  # Shape: [] (scalar)
        value = value.squeeze(0).squeeze(0)  # Shape: [] (scalar)

        return action, log_prob, value, next_hidden_state

    def rollout(self, env, buffer, rollout_steps):
        """
        Collects data from the environment for 'rollout_steps' and
        stores it in the buffer.

        Args:
            env (Environment): Your custom time-series environment.
            buffer (RolloutBuffer): The buffer to store data in.
            rollout_steps (int): Total steps to collect.

        Returns:
            Tuple of (last_obs, last_hidden_state, last_done)
            (Needed for GAE bootstrapping)
        """

        obs = env.reset()
        hidden_state = self.model.get_initial_states(batch_size=1, device=self.device)
        done = False

        for _ in range(rollout_steps):
            with torch.no_grad():

                # Format observation for model: [1, 1, input_dim]
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

                # Get action and other values from the model
                action, log_prob, value, next_hidden = self.get_action_and_value(obs_tensor, hidden_state)

            # Action must be a NumPy array for the environment
            action_np = action.cpu().numpy()

            # Take a step in the environment
            next_obs, reward, done, _ = env.step(action_np)

            # Store the transition in the buffer
            buffer.store(
                obs=obs,
                action=action_np,
                log_prob=log_prob.cpu().item(),
                value=value.cpu().item(),
                reward=reward,
                done=done
            )

            # Update states for next loop
            obs = next_obs
            hidden_state = next_hidden

            if done:
                # If episode ends, reset env and hidden state
                obs = env.reset()
                hidden_state = self.model.get_initial_states(batch_size=1, device=self.device)

        # Return the final states for GAE calculation
        return obs, hidden_state, done

    def update(self, buffer):
        """
        Performs the PPO update (learning) phase.
        This function is called *after* the buffer has been filled
        and advantages have been computed.

        Args:
            buffer (RolloutBuffer): The buffer containing all rollout data.

        Returns:
            dict: A dictionary of mean loss values for logging.
        """

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