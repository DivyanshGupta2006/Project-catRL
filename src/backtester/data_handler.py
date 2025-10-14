import pandas as pd
import json

from src.agent import predict
from src.risk_management import slippage, stop_loss, take_profit
from src.position_sizing import amount_calculator
from src.utils import get_config, read_file, get_absolute_path

config = get_config.read_yaml()

def fetch_data(data):
    state = read_file.read_state()
    dir = get_absolute_path.absolute(config['paths']['state_directory'])
    timestep = state['timestep']
    state['timestep'] = pd.to_datetime(state['timestep']) + pd.Timedelta(config['data']['timeframe'])
    with open(dir, 'w') as f:
        json.dump(state, f, indent=4)
    return data.loc[timestep]

def add_predictions(candle):
    candle = predict.predict_position(candle)
    candle = slippage.get_order_price(candle)
    candle = amount_calculator.calculate_amount(candle)
    candle = stop_loss.get_stop_loss(candle)
    candle = take_profit.get_take_profit(candle)
    return candle