import torch
import numpy as np

from src.utils import get_config

config = get_config.read_yaml()
num_assets = len(config["data"]["symbols"]) + 1

class Environment:
    def __init__(self, data):
        """
        Initializes the environment.

        Args:
            data (np.array): The full dataset (e.g., shape [N_steps, 126]).
                                         This data should be pre-normalized.
        """
        super(Environment, self).__init__()

        self.data = data
        self.max_steps = len(self.data) - 1  # Total steps in the data

        # Initialize state-tracking variables
        self.current_index = 0

        # --- Environment Configuration ---
        # These must match your model's dimensions

        # The observation is one row of your data
        self.observation_space_dim = 126

        # *** TODO: MUST BE MODIFIED BY YOU ***
        # This is an EXAMPLE.
        # 0 = Hold, 1 = Buy, 2 = Sell
        # If you have a different action scheme, change this.
        self.action_space_dim = 3

        print(f"Environment initialized with {self.max_steps + 1} timesteps.")

    def _calculate_reward(self, action):
        """
        Calculates the reward for taking a given action at the current timestep.

        This is the **CORE LOGIC** of your environment and is entirely
        dependent on your specific problem (e.g., trading, forecasting).

        --- *** TODO: REPLACE THIS WITH YOUR LOGIC *** ---

        Args:
            action (int): The action taken by the agent (e.g., 0, 1, or 2).

        Returns:
            float: The calculated reward.
        """

        # EXAMPLE LOGIC: TRADING
        # Let's assume:
        # - self.data[self.current_index][0] is the current price.
        # - self.data[self.current_index + 1][0] is the next day's price.
        # - Action 0: Hold (no reward)
        # - Action 1: Buy (reward is profit/loss from holding for one day)
        # - Action 2: Sell (reward is profit/loss from shorting for one day)

        reward = 0.0

        # Ensure we don't look past the end of the data
        if self.current_index >= self.max_steps:
            return 0.0

        price_now = self.data[self.current_index, 0]
        price_next = self.data[self.current_index + 1, 0]
        price_change = price_next - price_now

        if action == 1:  # Buy
            reward = price_change
        elif action == 2:  # Sell (Short)
            reward = -price_change
        # elif action == 0: # Hold
        #    reward = 0.0 # Or maybe a small penalty

        return reward
        # --- *** END OF EXAMPLE LOGIC *** ---

    def reset(self):
        """
        Resets the environment to the beginning of the time-series.

        Returns:
            np.array: The first observation (shape [126]).
        """
        # Reset the index to the start
        self.current_index = 0
        obs = self.data[self.current_index].copy()

        return obs

    def step(self, action):
        """
        Advances the environment by one timestep.

        Args:
            action (int): The action chosen by the agent.

        Returns:
            tuple: A 4-item tuple (next_observation, reward, done, info)
        """

        # 1. Calculate reward based on the action at the *current* step
        reward = self._calculate_reward(action)

        # 2. Move time forward
        self.current_index += 1

        # 3. Check if the episode is "done" (i.e., we ran out of data)
        # This is a TERMINATION, not a truncation.
        done = (self.current_index >= self.max_steps)

        # 4. Get the next observation
        if done:
            # If done, we can just return the last observation again
            # or a set of zeros. Returning the last one is fine.
            next_obs = self.data[self.max_steps].copy()
        else:
            next_obs = self.data[self.current_index].copy()

        # 5. Return the standard tuple
        # 'info' dictionary is typically empty in simple envs
        info_dict = {}

        return next_obs, reward, done, info_dict
