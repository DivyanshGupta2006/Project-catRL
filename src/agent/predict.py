from src.backtester import execute_SL_TP
from src.utils import get_config

config = get_config.read_yaml()

def predict_position(candle):
    cryptos = list({token for (_, token) in candle.index})
    for crypto in cryptos:
        candle[('fiducia',crypto)] = 0.10
    return candle