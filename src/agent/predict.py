from src.utils import get_config

config = get_config.read_yaml()

def predict_position(candle):
    for crypto in candle.keys():
        candle[crypto]['fiducia'] = 0.10
    return candle