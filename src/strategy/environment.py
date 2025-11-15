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

        # --- 1. GET CURRENT STATE (Time t) ---
        state_t = read_file.read_state()
        t = state_t['timestep']

        # Get the raw data row for time t
        row_t = self.data.loc[self.data.index[t]]

        # Convert row to the 'candle' dict format
        candle_t = self._row_to_candle_dict(row_t)

        # --- 2. EXECUTE PRE-TRADE LOGIC ---
        # Execute any stop-loss / take-profit orders from *previous* step
        execute_SL_TP.execute(candle_t)

        # Calculate portfolio value *before* new trades
        value_t = portfolio_calculator.calculate(candle_t)

        # --- 3. APPLY AGENT'S ACTION (Time t) ---
        # Assign the agent's action (fiduciae) to the candle
        candle_t = self._assign_fiduciae(candle_t, fiduciae_action)

        # --- 4. EXECUTE TRADING LOGIC (Time t) ---
        # Run the full backtesting pipeline using the agent's fiduciae
        candle_t = slippage.get_order_price(candle_t, value_t)
        candle_t = amount_calculator.calculate(candle_t, value_t)
        candle_t = stop_loss.get_stop_loss(candle_t)
        candle_t = take_profit.get_take_profit(candle_t)

        order = place_order.place(candle_t)

        # This function updates portfolio.csv and state.json (cash)
        execute_order.execute(order)

        # --- 5. ADVANCE TIME & GET REWARD (Time t+1) ---
        # Manually advance the timestep in state.json
        state_t['timestep'] += 1
        update_state.update(state_t)

        t_plus_1 = state_t['timestep']

        # Check if the episode is done
        done = (t_plus_1 >= self.max_steps)

        # Get the index for the next observation
        # If done, we just use the last available step
        next_obs_index = self.max_steps - 1 if done else t_plus_1

        # Get the raw data row for time t+1
        row_t_plus_1 = self.data.loc[self.data.index[next_obs_index]]

        # Convert to candle dict to calculate the new portfolio value
        candle_t_plus_1 = self._row_to_candle_dict(row_t_plus_1)

        # Calculate the new portfolio value at time t+1
        value_t_plus_1 = portfolio_calculator.calculate(candle_t_plus_1)

        # --- 6. CALCULATE REWARD ---
        # The reward is the change in portfolio value
        reward = value_t_plus_1 - value_t

        # Get the next observation to return to the agent
        next_obs = row_t_plus_1.to_numpy()

        info_dict = {}  # Placeholder for extra info

        return next_obs, reward, done, info_dict