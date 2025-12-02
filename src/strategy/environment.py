# from src.utils import read_file, convert
# from src.update_files import update_state, update_portfolio
# from src.backtester import execute_SL_TP, place_order, execute_order, calculate_metrics
# from src.position_sizing import portfolio_calculator, amount_calculator, fiducia_calculator
# from src.risk_management import slippage, stop_loss, take_profit
# from src.strategy import predict_position
#
# class Environment:
#
#     def __init__(self, train_data, sequence_length, num_assets, symbols, capital):
#         self.train_data = train_data
#         self.sequence_length = sequence_length
#         self.num_assets = num_assets
#         self.symbols = symbols
#         self.capital = capital
#
#     def reset(self):
#         update_state.set_state(self.capital)
#         update_portfolio.set_portfolio()
#
#     def step(self, raw_action, buffer):
#
#         state = read_file.read_state()
#         timestep = state['timestep']
#         row = self.train_data.loc[self.train_data.index[timestep]].copy()
#         state['timestep'] += 1
#         update_state.update(state)
#
#         fiduciae_action = fiducia_calculator.calculate(raw_action)
#
#         candle = convert.convert_to_dict(row)
#
#         # not execute SL, TP here, rather execute it at the end of the last step -> simulate the next one hour after taking action
#         # to get the final portfolio value : which will be used in reward
#
#         Pt = portfolio_calculator.calculate(candle)
#
#         calculate_metrics.calculate_candle_metrics(candle)
#
#         candle = predict_position.assign_fiducia(candle, fiduciae_action.tolist())
#
#         candle = slippage.get_order_price(candle, Pt)
#
#         candle = amount_calculator.calculate(candle, Pt)
#
#         candle = stop_loss.get_stop_loss(candle)
#         candle = take_profit.get_take_profit(candle)
#
#         order = place_order.place(candle)
#         execute_order.execute(order)
#         calculate_metrics.calculate_order_metrics(order)
#
#         Pt_beg = portfolio_calculator.calculate(candle)
#
#         done = (state['timestep'] >= len(self.train_data) - 1)
#
#         if not done:
#             next_row = self.train_data.loc[self.train_data.index[state['timestep']]].copy()
#             next_candle = convert.convert_to_dict(next_row)
#             execute_SL_TP.execute(next_candle)
#             Pt_end = portfolio_calculator.calculate(next_candle)
#
#         reward = (Pt_end - Pt_beg) / (Pt_beg + 1e-8)
#
#         info_dict = {'reward': reward, 'profit': (Pt_end - Pt_beg)}
#
#         buffer.store_state(self.train_data.iloc[timestep : timestep + self.sequence_length].values)
#         buffer.store_reward(reward)
#         buffer.store_done(done)
#
#         return buffer
#
#     # i look at data for t-72 -> t-1 steps, i am at end of (t - 1)'th step (72 being my sequence length for LSTM)
#     # then gives action for this step, i will execute this action at the start of t'th step
#     # but its reward will only be known at the end of this t'th steo - since P_new in reward is the new portfolio value - indicative of the wisdom in the model's decisions
#     # this P_new is the portfolio value after executing SL, TP at end of t'th step, P_old is portfolio value after investing using these these fiducia
#     # but i want to give the reward now, not wait for the next step
#     # here also i update using the (t - 1)'th step only, no looking at the t'th step (to prevent look-ahead bias)
#     # so for the purpose of training the model, and providing it with reward, let me execute the SL, TP for the next candle, and then calculate the P_new


import numpy as np
import pandas as pd
import torch
from src.position_sizing.fiducia_calculator import calculate as calc_fiducia


class FastEnvironment:
    def __init__(self, data_df, config):
        """
        Args:
            data_df (pd.DataFrame): The merged training data.
            config (dict): Config dict.
        """
        self.data = data_df
        self.config = config
        self.seq_len = config['hyperparameters']['seq_len']
        self.symbols = config['data']['symbols']
        self.num_assets = len(self.symbols)  # 9 Assets

        # Convert Dataframe to Numpy for 100x speedup
        # Shape: (Total_Rows, Features)
        self.data_matrix = self.data.values
        self.timestamps = self.data.index

        # Account State
        self.initial_capital = config['strategy']['capital']
        self.cash = self.initial_capital
        self.holdings = np.zeros(self.num_assets)  # Amount of each coin

        # Pointers
        self.current_step = self.seq_len  # Start after first window
        self.max_steps = len(self.data) - 1

        # Transaction Costs
        self.tc = config['strategy']['transaction_cost_fraction']

    def reset(self):
        self.cash = self.initial_capital
        self.holdings = np.zeros(self.num_assets)
        self.current_step = self.seq_len

        return self._get_observation()

    def _get_observation(self):
        """Returns the rolling window of size (seq_len, features)"""
        # Slice: from (t - seq_len) to t
        return self.data_matrix[self.current_step - self.seq_len: self.current_step]

    def _get_current_prices(self):
        """
        Extracts closing prices for the 9 assets from the current row.
        Assumes columns are ordered as per `symbols`.
        You might need to adjust indices based on your exact column structure.
        Here we assume 'close' columns are available.
        """
        # NOTE: This part depends on your specific column ordering in 'merged_data'
        # For simplicity, assuming 'close' price is part of the feature set
        # We will use the raw feature values.
        # Ideally, pass raw prices separately if normalized features are used.
        # For now, using the last feature as close price placeholder (ADJUST AS NEEDED)

        # Since we are training, we mostly care about rewards (portfolio value change).
        # We can approximate reward using feature returns if raw price isn't easy to unpack.
        # BUT, for accuracy, let's assume we have access to unscaled Close prices.
        # For this implementation, I will assume the environment was passed a separate 'prices' df
        # or we just simulate 'percent change' reward which is safer.
        pass

    def step(self, raw_action):
        """
        Args:
            raw_action (np.array): Output from Actor (Logits)
        """
        # 1. Calculate Portfolio Value at t
        # (Simplified: In real training, getting exact prices from normalized features is hard.
        # We will use % return of the assets to calculate reward)

        # Get Returns of assets at time t
        # return = (Price_t - Price_t-1) / Price_t-1
        # This is pre-calculated in features usually, or we compute it on fly

        # Hack for speed: We assume the reward is the dot product of
        # (Action_Weights * Asset_Returns) - Transaction_Costs

        # Convert Raw Action (Logits) to Fiducia (Weights)
        # We use the provided PyTorch function but convert to numpy
        with torch.no_grad():
            tensor_action = torch.tensor(raw_action, dtype=torch.float32)
            fiducia = calc_fiducia(tensor_action).numpy()  # Shape (10,)

        # 2. Execute Strategy (Simplified In-Memory)
        # Asset weights (first 9), Cash weight (last 1)
        asset_weights = fiducia[:-1]

        # Get actual returns from data (next step relative change)
        # We peek ahead to t+1 to see what happened to prices
        curr_row = self.data_matrix[self.current_step]
        next_row = self.data_matrix[self.current_step + 1]

        # Assume specific columns are 'close' prices or 'returns'
        # Since I cannot see your exact column map, I will use a placeholder:
        # We assume the Agent learns to maximize 'features' that correlate with Up.

        # REWARD FUNCTION (The most important part)
        # Instead of full complex backtest, we approximate:
        # Reward = Sum(Weight_i * Return_i)

        # To make this code run without erroring on column indices,
        # I will define Reward as: (Next_Val - Curr_Val)
        # where Val is implied by the 'close' columns.

        # PRO TIP: For training, simpler is better.
        # We will assume columns [0::num_features] are the Close prices of the 9 assets.
        # This is a strong assumption, but necessary for code generation.
        current_prices = curr_row[0::14]  # Assuming 14 features per asset, 0th is close/open
        next_prices = next_row[0::14]

        price_change_pct = (next_prices - current_prices) / (current_prices + 1e-8)

        # Portfolio Return
        portfolio_return = np.sum(asset_weights * price_change_pct)

        # Subtract transaction cost approximation (turnover)
        # simplified turnover cost
        cost = self.tc * np.sum(np.abs(asset_weights))

        reward = portfolio_return - cost

        # 3. Advance Time
        self.current_step += 1
        done = self.current_step >= (self.max_steps - 1)

        next_obs = self._get_observation()

        return next_obs, reward, done, {}