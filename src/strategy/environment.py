import numpy as np
import pandas as pd
import ast

from src.utils import get_config, read_file
from src.update_files import update_state, update_portfolio

from src.backtester import execute_SL_TP, place_order, execute_order
from src.position_sizing import portfolio_calculator, amount_calculator
from src.risk_management import slippage, stop_loss, take_profit

config = get_config.read_yaml()
NUM_ASSETS = len(config['data']['symbols']) + 1  # 9 cryptos + 1 cash


class Environment:

    def __init__(self, data):
        """
        Initializes the environment.

        Args:
            data (pd.DataFrame): The full, merged, and preprocessed
                                             time-series data.
        """
        # super(Environment, self).__init__()

        self.data = data
        self.max_steps = len(self.data) - 1

        # --- Environment Configuration ---
        self.action_space_dim = NUM_ASSETS  # 10 (9 cryptos + 1 cash)
        self.observation_space_dim = len(self.data.columns)  # 126

        # --- Helper attributes from config ---
        self.config = config
        self.symbols = self.config['data']['symbols']
        # Get 'ETH' from 'ETH/USDT'
        self.crypto_symbols = [s.split('/')[0] for s in self.symbols]

        print(f"Environment initialized with {self.max_steps + 1} timesteps.")
        print(f"Observation space dim: {self.observation_space_dim}")
        print(f"Action space dim: {self.action_space_dim}")

    def _row_to_candle_dict(self, row):
        """
        Converts a single row from the merged DataFrame into the
        'candle' dictionary format required by the backtester scripts.
        This mimics the logic from 'data_handler.py'.
        """
        # The DataFrame's columns are string-represented tuples, e.g., "('close', 'ETH')"
        # We must convert them to actual tuples to unstack.
        try:
            row.index = row.index.map(ast.literal_eval)
        except ValueError:
            # If index is already evaluated, skip
            pass

        candle_df = row.unstack(level=0)
        candle = candle_df.to_dict(orient='index')
        return candle

    def _assign_fiduciae(self, candle, fiduciae_action):
        """
        Assigns the agent's action (fiduciae vector) to the candle dict.
        This replaces the 'predict_position.py' script.
        """
        # fiduciae_action is a [10,] array
        # First 9 are for cryptos, 10th is for cash
        for i, symbol in enumerate(self.crypto_symbols):
            if symbol in candle:
                # Assign the fiduciae (weight) for this crypto
                candle[symbol]['fiducia'] = fiduciae_action[i]

        # Note: The 10th action (cash fiducia) is implicitly handled
        # by the position sizing and order execution logic.
        return candle

    def reset(self):
        """
        Resets the environment state to the very beginning.
        - Resets the state.json (timestep=0, cash=initial)
        - Resets the portfolio.csv (all assets=0)
        - Returns the first observation.
        """
        # Set state.json to timestep 0 and initial capital
        update_state.set_state(self.config['strategy']['capital'])

        # Set portfolio.csv to all zeros
        update_portfolio.set_portfolio()

        # Get the very first observation (row 0)
        # .loc is used because the index is a timestamp
        obs = self.data.loc[self.data.index[0]].to_numpy()

        return obs

    def step(self, fiduciae_action):
        """
        Advances the environment by one timestep using the agent's action.

        Args:
            fiduciae_action (np.array): The vector of 10 continuous actions
                                        (fiduciae/weights) from the agent.

        Returns:
            tuple: (next_observation, reward, done, info)
        """

        # get the current timestep

        # get the row from merged_training_data

        # execute SL, TP

        # metrics

        # portfolio size

        # get order price using slippage

        # amount calculator

        # stop loss, take profit calculate

        # order metrics

        # calculate reward

        # pick up required data from merged data

        # normalize to prepare next_obs

        # info_dict

        return next_obs, reward, done, info_dict