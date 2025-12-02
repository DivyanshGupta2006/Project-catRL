# # from src.utils import get_config
# #
# # config = get_config.read_yaml()
# #
# # symbols = config['data']['symbols']
# # symbols = [symbol.split('/')[0] for symbol in symbols]
# #
# # def predict_fiduciae(candle):
# #     return [0.92, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
# #
# # def assign_fiducia(candle, fiduciae):
# #     for idx,crypto in enumerate(symbols):
# #         candle[crypto]['fiducia'] = fiduciae[idx]
# #
# #     return candle
# #
# # def predict(candle):
# #     fiduciae = predict_fiduciae(candle)
# #     candle = assign_fiducia(candle, fiduciae)
# #     return candle
#
# import torch
# import numpy as np
# from collections import deque
# import os
#
# from src.strategy.model import Model
# from src.position_sizing import fiducia_calculator
# from src.utils import get_config, get_absolute_path
#
#
# # --- Global State Holder ---
# class Predictor:
#     def __init__(self):
#         self.config = get_config.read_yaml()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         # 1. Load Model Architecture
#         hp = self.config['hyperparameters']
#         self.seq_len = hp['seq_len']  # 72
#         self.input_dim = hp['input_dim']
#
#         self.model = Model(
#             input_dim=self.input_dim,
#             lstm_hidden_dim=hp['hidden_state_dim'],
#             n_assets=hp['num_assets'],
#             n_lstm_layers=hp['num_lstm_layers'],
#             actor_hidden_dim=hp['actor_hidden_dim'],
#             critic_hidden_dim=hp['critic_hidden_dim']
#         ).to(self.device)
#
#         # 2. Load Weights
#         model_path = get_absolute_path.absolute(self.config['paths']['model_directory'] + 'ppo_agent.pth')
#         if os.path.exists(model_path):
#             self.model.load_state_dict(torch.load(model_path, map_location=self.device))
#             self.model.eval()
#             print("--- Model Loaded Successfully for Inference ---")
#         else:
#             print(f"WARNING: Model not found at {model_path}. Using random initialization.")
#
#         # 3. Initialize Rolling Buffer
#         # We need a history of 'seq_len' steps.
#         self.buffer = deque(maxlen=self.seq_len)
#
#         # Define Feature Order exactly as it was during training (Data Processing)
#         # Note: This MUST match the order in feature_engineer.py / raw data
#         self.feature_list = [
#             'open', 'high', 'low', 'close', 'volume',
#             'rsi',
#             'sma-50', 'sma-100', 'sma-200',
#             'ema-50', 'ema-100', 'ema-200',
#             'atr', 'adx'
#         ]
#         self.symbols = self.config['data']['symbols']
#         # Extract base symbols like 'ETH', 'BTC' from 'ETH/USDT'
#         self.crypto_keys = [s.split('/')[0] for s in self.symbols]
#
#     def _candle_to_vector(self, candle):
#         """
#         Flattens the candle dictionary into a feature vector (numpy array).
#         Order: Symbol1_Feats -> Symbol2_Feats ...
#         """
#         vector = []
#         for symbol in self.crypto_keys:
#             # Check if symbol exists in candle (it should)
#             if symbol in candle:
#                 data_point = candle[symbol]
#                 # Extract features in strict order
#                 for feat in self.feature_list:
#                     # In case of missing keys (e.g. at very start), handle gracefully usually 0
#                     val = data_point.get(feat, 0.0)
#                     vector.append(val)
#         return np.array(vector, dtype=np.float32)
#
#     def predict_step(self, candle):
#         # 1. Process Input
#         current_vector = self._candle_to_vector(candle)
#
#         # 2. Update Buffer
#         self.buffer.append(current_vector)
#
#         # 3. Check for Warm-up
#         if len(self.buffer) < self.seq_len:
#             # Not enough data yet to fill the LSTM window.
#             # Return neutral/hold fiduciae (0.0) or handle as needed.
#             # Here we just assign 0.0 fiducia to wait.
#             for crypto in candle:
#                 candle[crypto]['fiducia'] = 0.0
#             return candle
#
#         # 4. Prepare Tensor
#         # Shape: (1, 72, Input_Dim)
#         seq_array = np.array(self.buffer)
#         state_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(self.device)
#
#         # 5. Inference
#         with torch.no_grad():
#             # Model returns (Dist, Value). We need the action mean (or sample).
#             # Usually for inference/backtest we use the Mean (Deterministic)
#             # OR Sample (Stochastic).
#             # Let's use the Mean for stable backtesting.
#             dist, _ = self.model(state_tensor)
#
#             # Get the mean (loc) of the Normal distribution
#             # dist is a Normal(loc, scale) object
#             raw_action = dist.loc
#
#             # Calculate final fiduciae (Softmax logic)
#             fiduciae = fiducia_calculator.calculate(raw_action)
#             fiduciae = fiduciae.cpu().numpy().flatten()
#
#         # 6. Assign to Candle
#         # fiduciae has 10 values (9 cryptos + 1 cash)
#         for i, symbol in enumerate(self.crypto_keys):
#             if symbol in candle:
#                 candle[symbol]['fiducia'] = float(fiduciae[i])
#
#         return candle
#
#
# # --- Instantiate Singleton ---
# # This runs once when the module is imported
# _predictor = Predictor()
#
#
# def predict(candle):
#     """
#     Main entry point called by backtest_strategy.py
#     """
#     return _predictor.predict_step(candle)


import torch
import numpy as np
from collections import deque
import os

from src.strategy.model import Model
from src.position_sizing import fiducia_calculator
from src.utils import get_config, get_absolute_path


# --- Global State Holder ---
class Predictor:
    def __init__(self):
        self.config = get_config.read_yaml()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load Model Architecture
        hp = self.config['hyperparameters']
        self.seq_len = hp['seq_len']  # 72
        self.input_dim = hp['input_dim']

        self.model = Model(
            input_dim=self.input_dim,
            lstm_hidden_dim=hp['hidden_state_dim'],
            n_assets=hp['num_assets'],
            n_lstm_layers=hp['num_lstm_layers'],
            actor_hidden_dim=hp['actor_hidden_dim'],
            critic_hidden_dim=hp['critic_hidden_dim']
        ).to(self.device)

        # 2. Load Weights
        model_path = get_absolute_path.absolute(self.config['paths']['model_directory'] + 'ppo_agent.pth')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("--- Model Loaded Successfully for Inference ---")
        else:
            print(f"WARNING: Model not found at {model_path}. Using random initialization.")

        # 3. Initialize Rolling Buffer
        # We need a history of 'seq_len' steps.
        self.buffer = deque(maxlen=self.seq_len)

        # Define Feature Order exactly as it was during training (Data Processing)
        # Note: This MUST match the order in feature_engineer.py / raw data
        self.feature_list = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi',
            'sma-50', 'sma-100', 'sma-200',
            'ema-50', 'ema-100', 'ema-200',
            'atr', 'adx'
        ]
        self.symbols = self.config['data']['symbols']
        # Extract base symbols like 'ETH', 'BTC' from 'ETH/USDT'
        self.crypto_keys = [s.split('/')[0] for s in self.symbols]

    def _candle_to_vector(self, candle):
        """
        Flattens the candle dictionary into a feature vector (numpy array).
        Order: Symbol1_Feats -> Symbol2_Feats ...
        """
        vector = []
        for symbol in self.crypto_keys:
            # Check if symbol exists in candle (it should)
            if symbol in candle:
                data_point = candle[symbol]
                # Extract features in strict order
                for feat in self.feature_list:
                    # In case of missing keys (e.g. at very start), handle gracefully usually 0
                    val = data_point.get(feat, 0.0)
                    vector.append(val)
        return np.array(vector, dtype=np.float32)

    def predict_step(self, candle):
        # 1. Process Input
        current_vector = self._candle_to_vector(candle)

        # 2. Update Buffer
        self.buffer.append(current_vector)

        # 3. Check for Warm-up
        if len(self.buffer) < self.seq_len:
            # Not enough data yet to fill the LSTM window.
            # Return neutral/hold fiduciae (0.0) or handle as needed.
            # Here we just assign 0.0 fiducia to wait.
            for crypto in candle:
                candle[crypto]['fiducia'] = 0.1
            return candle

        # 4. Prepare Tensor
        # Shape: (1, 72, Input_Dim)
        seq_array = np.array(self.buffer)
        state_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 5. Inference
        with torch.no_grad():
            # Model returns (Dist, Value). We need the action mean (or sample).
            # Usually for inference/backtest we use the Mean (Deterministic)
            # OR Sample (Stochastic).
            # Let's use the Mean for stable backtesting.
            dist, _ = self.model(state_tensor)

            # Get the mean (loc) of the Normal distribution
            # dist is a Normal(loc, scale) object
            raw_action = dist.loc

            # Calculate final fiduciae (Softmax logic)
            fiduciae = fiducia_calculator.calculate(raw_action)
            fiduciae = fiduciae.cpu().numpy().flatten()

        # 6. Assign to Candle
        # fiduciae has 10 values (9 cryptos + 1 cash)
        for i, symbol in enumerate(self.crypto_keys):
            if symbol in candle:
                candle[symbol]['fiducia'] = float(fiduciae[i])

        return candle


# --- Instantiate Singleton ---
# This runs once when the module is imported
_predictor = Predictor()


def predict(candle):
    """
    Main entry point called by backtest_strategy.py
    """
    return _predictor.predict_step(candle)