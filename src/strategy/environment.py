import torch
import numpy as np

class Environment:

    def __init__(self, n_assets, seq_len, input_dim):
        self.n_assets = n_assets
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.current_step = 0

        # Your context: 4 years of 1-hour data
        self.total_steps = 35000
        print("Initializing Placeholder Trading Environment...")

        # Simulate loading data and calculating features
        # In your real code, you'd load your processed data here
        # This dummy data has shape (TotalSteps, NumFeatures * NumAssets)
        # We use n_assets-1 (9) for the input features
        n_input_assets = n_assets - 1
        n_features = input_dim // n_input_assets

        print(f"Env Setup: {n_assets} total assets ({n_input_assets} cryptos + 1 cash)")
        print(f"Env Setup: {n_features} features per crypto asset")
        print(f"Env Setup: {input_dim} total input features")

        # This self.full_state_data simulates your preprocessed data file
        self.full_state_data = torch.randn(self.total_steps, self.input_dim)

        # Simulate portfolio value
        self.portfolio_value = 10000.0  # Initial portfolio

    def _get_state(self):
        """
        Gets the state for the current time step.
        The state is a window of the last `seq_len` observations.
        """
        if self.current_step < self.seq_len:
            # Not enough history, pad with zeros
            state = torch.zeros(self.seq_len, self.input_dim)
            state[self.seq_len - self.current_step:] = self.full_state_data[:self.current_step]
        else:
            state = self.full_state_data[self.current_step - self.seq_len: self.current_step]

        # Add a batch dimension for the model
        return state.unsqueeze(0)  # Shape: (1, seq_len, input_dim)

    def reset(self):
        """
        Resets the environment and returns the initial state.
        State should be a torch tensor of shape (1, seq_len, input_dim)
        """
        print("Environment reset.")
        self.current_step = 0
        self.portfolio_value = 10000.0
        return self._get_state()

    def step(self, portfolio_weights):
        """
        Takes an action (portfolio_weights) and returns the next state,
        reward, done, and info.

        Args:
            portfolio_weights (torch.Tensor): 
                Shape (10,) -> 9 assets + 1 cash

        Returns:
            next_state (torch.Tensor): Shape (1, seq_len, input_dim)
            reward (float): The reward (P_t - P_t-1)
            done (bool): True if the episode is over
            info (dict): {}
        """
        if self.current_step >= self.total_steps - 1:
            # End of data
            self.current_step += 1
            return self._get_state(), 0, True, {}

        # --- Simulate portfolio value change ---
        # This is where your backtester logic would go.
        # For this placeholder, we'll simulate a reward.
        prev_portfolio_value = self.portfolio_value

        # Simulate market movement - biased towards positive for "progress"
        # and tied to how much is *not* in cash (risk-on)
        cash_weight = portfolio_weights[-1].item()
        risk_on_weight = 1.0 - cash_weight
        market_return = (np.random.rand() * 0.02 - 0.008)  # -0.8% to +1.2%
        reward = (market_return * risk_on_weight) * self.portfolio_value

        # Your reward function
        # reward = P_t - P_t-1
        self.portfolio_value += reward

        # Move to the next time step
        self.current_step += 1

        # Get the next state
        next_state = self._get_state()

        # Check if done
        done = self.current_step >= self.total_steps - 1

        if self.current_step % 1000 == 0:
            print(f"Env Step: {self.current_step}/{self.total_steps} | Portfolio: ${self.portfolio_value:.2f}")

        return next_state, reward, done, {}