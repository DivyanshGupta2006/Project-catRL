import torch
import numpy as np
from collections import deque

from src.strategy.model import Model
from src.position_sizing import fiducia_calculator
from src.utils import get_config, get_absolute_path, read_file, convert

config = get_config.read_yaml()


class Predictor:

    def __init__(self,
                 data,
                 n_assets,
                 input_dim,
                 lstm_hidden_dim,
                 num_lstm_layers,
                 actor_hidden_dim,
                 critic_hidden_dim,
                 seq_len,
                 symbols,
                 model_path,
                 device):
        self.data = data
        self.n_assets = n_assets
        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.seq_len = seq_len
        self.symbols = symbols
        self.model_path = model_path
        self.device = device

        self.model = Model(self.n_assets,
                           self.input_dim,
                           self.lstm_hidden_dim,
                           self.num_lstm_layers,
                           self.actor_hidden_dim,
                           self.critic_hidden_dim).to(self.device)

        self.model_loaded = False

        self.buffer = deque(maxlen=seq_len)

        self.feature_list = [
            'volume',
            'rsi', 'mfi', 'adx', 'bop',
            'stochastic_oscillator-slow', 'macd-line', 'macd-hist',
            'sma-200', 'ema-50', 'tema-50',
            'obv', 'atr', 'candle'
        ]

        for idx, symbol in enumerate(self.symbols):
            self.symbols[idx] = symbol.split('/')[0]

    def load_model(self):
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.model_loaded = True
        except:
            print(f'Failed to load model')
            self.model_loaded = False

    def _candle_to_vector(self, candle):
        vector = []
        for symbol in self.symbols:
            data = candle[symbol]
            for feature in self.feature_list:
                val = data.get(feature, 0.0)
                vector.append(val)

        return np.array(vector, dtype=np.float32)

    def assign_prediction(self, candle):
        if not self.model_loaded:
            self.load_model()

        state = read_file.read_state()

        row = self.data.iloc[state['timestep']]

        candle_ = convert.convert_to_dict(row)

        current_vector = self._candle_to_vector(candle_)

        self.buffer.append(current_vector)

        if len(self.buffer) < self.seq_len:
            for crypto in candle:
                candle[crypto]['fiducia'] = 0.1

            return candle

        sequence = np.array(self.buffer)
        state = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist, _ = self.model.forward(state)
            raw_action = dist.loc

            fiduciae = fiducia_calculator.calculate(raw_action)
            fiduciae = fiduciae.cpu().numpy().flatten()

        for i, symbol in enumerate(self.symbols):
            candle[symbol]['fiducia'] = float(fiduciae[i])

        return candle


data = None

hp = config['hyperparameters']

NUM_ASSETS = hp['num_assets']
INPUT_DIM = hp['input_dim']
LSTM_HIDDEN_DIM = hp['hidden_state_dim']
NUM_LSTM_LAYERS = hp['num_lstm_layers']
ACTOR_HIDDEN_DIM = hp['actor_hidden_dim']
CRITIC_HIDDEN_DIM = hp['critic_hidden_dim']
SEQUENCE_LENGTH = hp['seq_len']

symbols = config['data']['symbols']
model_path = get_absolute_path.absolute(config['paths']['model_directory'] + 'model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

predictor = Predictor(data,
                      NUM_ASSETS,
                      INPUT_DIM,
                      LSTM_HIDDEN_DIM,
                      NUM_LSTM_LAYERS,
                      ACTOR_HIDDEN_DIM,
                      CRITIC_HIDDEN_DIM,
                      SEQUENCE_LENGTH,
                      symbols,
                      model_path,
                      device)

def assign_field_of_view(field_of_view):
    global data
    global predictor
    data = field_of_view

    hp = config['hyperparameters']

    NUM_ASSETS = hp['num_assets']
    INPUT_DIM = hp['input_dim']
    LSTM_HIDDEN_DIM = hp['hidden_state_dim']
    NUM_LSTM_LAYERS = hp['num_lstm_layers']
    ACTOR_HIDDEN_DIM = hp['actor_hidden_dim']
    CRITIC_HIDDEN_DIM = hp['critic_hidden_dim']
    SEQUENCE_LENGTH = hp['seq_len']

    symbols = config['data']['symbols']
    model_path = get_absolute_path.absolute(config['paths']['model_directory'] + 'model.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predictor = Predictor(data,
                          NUM_ASSETS,
                          INPUT_DIM,
                          LSTM_HIDDEN_DIM,
                          NUM_LSTM_LAYERS,
                          ACTOR_HIDDEN_DIM,
                          CRITIC_HIDDEN_DIM,
                          SEQUENCE_LENGTH,
                          symbols,
                          model_path,
                          device)

def predict(candle):
    global predictor
    return predictor.assign_prediction(candle)
