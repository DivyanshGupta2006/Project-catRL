# import torch
#
# class Buffer:
#     def __init__(self):
#         self.states = []
#         self.actions = []
#         self.log_probs = []
#         self.values = []
#         self.rewards = []
#         self.entropies = []
#         self.advantages = []
#         self.returns = []
#         self.dones = []
#
#     def append_data(self, state, action, log_prob, value, reward, entropy, advantage, _return, done):
#         self.states.append(state)
#         self.actions.append(action)
#         self.log_probs.append(log_prob)
#         self.values.append(value)
#         self.rewards.append(reward)
#         self.entropies.append(entropy)
#         self.advantages.append(advantage)
#         self.returns.append(_return)
#         self.dones.append(done)
#
#     def clear(self):
#         self.states.clear()
#         self.actions.clear()
#         self.log_probs.clear()
#         self.values.clear()
#         self.rewards.clear()
#         self.entropies.clear()
#         self.advantages.clear()
#         self.returns.clear()
#         self.dones.clear()
#
#     def display(self):
#         states = torch.tensor(self.states)
#         actions = torch.tensor(self.actions)
#         log_probs = torch.tensor(self.log_probs)
#         values = torch.tensor(self.values)
#         rewards = torch.tensor(self.rewards)
#         entropies = torch.tensor(self.entropies)
#         advantages = torch.tensor(self.advantages)
#         returns = torch.tensor(self.returns)
#         dones = torch.tensor(self.dones)
#
#         print("states:", states, states.shape)
#         print("actions:", actions, actions.shape)
#         print("log_probs:", log_probs, log_probs.shape)
#         print("values:", values, values.shape)
#         print("rewards:", rewards, rewards.shape)
#         print("entropies:", entropies, entropies.shape)
#         print("advantages:", advantages, advantages.shape)
#         print("returns:", returns, returns.shape)
#         print("dones:", dones, dones.shape)


import torch
import numpy as np


class Buffer:
    """
    A buffer for storing trajectories (rollouts) for Recurrent PPO.
    It stores data flatly but samples it in sequences.
    """

    def __init__(self, capacity):
        """
        Args:
            capacity (int): The total number of steps to store (ROLLOUT_STEPS).
        """
        self.capacity = capacity

        # We will store data as flat lists
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

        # These will be calculated *after* the rollout
        self.advantages = []
        self.returns = []

    def store(self, obs, action, log_prob, value, reward, done):
        """
        Stores a single timestep of experience.

        Args:
            obs (np.array): The observation.
            action (np.array): The action taken (shape [10,]).
            log_prob (float): The log-probability of the action.
            value (float): The value estimate from the critic.
            reward (float): The reward received.
            done (bool): Whether the episode terminated.
        """
        if len(self.observations) < self.capacity:
            self.observations.append(obs)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.rewards.append(reward)
            self.dones.append(done)

    def clear(self):
        """Clears all stored data."""
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.advantages.clear()
        self.returns.clear()

    def compute_advantages(self, gamma, gae_lambda, last_value, last_done):
        """
        Computes advantages and returns-to-go (GAE) for the entire buffer.
        This must be called *after* the rollout is complete.

        Args:
            gamma (float): Discount factor.
            gae_lambda (float): GAE smoothing parameter.
            last_value (float): Value estimate of the *final* observation.
            last_done (bool): 'done' flag of the *final* observation.
        """

        # We need to process data in reverse
        n_steps = len(self.observations)

        # Initialize advantages/returns with zeros
        self.advantages = [0.0] * n_steps
        self.returns = [0.0] * n_steps

        # GAE calculation
        last_gae_lam = 0
        for t in reversed(range(n_steps)):
            # If this is the last step, use the 'last_value' for bootstrapping
            if t == n_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            # GAE Delta
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]

            # GAE Advantage
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

            self.advantages[t] = last_gae_lam

        # Calculate returns (Returns = Advantages + Values)
        self.returns = [adv + val for adv, val in zip(self.advantages, self.values)]

    def sample(self, batch_size, sequence_length):
        """
        A generator that yields mini-batches of sequences.

        Args:
            batch_size (int): Num of sequences per batch (BATCH_SIZE).
            sequence_length (int): Length of each sequence (SEQUENCE_LENGTH).

        Yields:
            A tuple of batched tensors (obs, actions, log_probs, advantages, returns)
        """

        n_steps = len(self.observations)

        # Calculate number of valid starting points for a sequence
        n_valid_starts = n_steps - sequence_length
        if n_valid_starts <= 0:
            raise ValueError(f"Not enough data ({n_steps} steps) to sample "
                             f"sequences of length {sequence_length}.")

        # Total number of timesteps we'll process per epoch
        total_timesteps = n_valid_starts * self.capacity // n_steps

        # Calculate number of mini-batches
        n_mini_batches = total_timesteps // (batch_size * sequence_length)
        if n_mini_batches == 0:
            n_mini_batches = 1  # At least one batch

        for _ in range(n_mini_batches):
            # 1. Generate random starting indices
            start_indices = np.random.randint(0, n_valid_starts, size=batch_size)

            # 2. Build the batches by slicing
            batch_obs = []
            batch_actions = []
            batch_log_probs = []
            batch_advantages = []
            batch_returns = []

            for start in start_indices:
                end = start + sequence_length

                batch_obs.append(self.observations[start:end])
                batch_actions.append(self.actions[start:end])
                batch_log_probs.append(self.log_probs[start:end])
                batch_advantages.append(self.advantages[start:end])
                batch_returns.append(self.returns[start:end])

            # 3. Convert lists of sequences to batched tensors
            obs_tensor = torch.tensor(np.array(batch_obs), dtype=torch.float32)
            actions_tensor = torch.tensor(np.array(batch_actions), dtype=torch.float32)
            log_probs_tensor = torch.tensor(np.array(batch_log_probs), dtype=torch.float32)
            advantages_tensor = torch.tensor(np.array(batch_advantages), dtype=torch.float32)
            returns_tensor = torch.tensor(np.array(batch_returns), dtype=torch.float32)

            yield (obs_tensor, actions_tensor, log_probs_tensor,
                   advantages_tensor, returns_tensor)