from src.utils import get_config

config = get_config.read_yaml()

def predict(candle):
    for crypto in candle:
        candle[crypto]['fiducia'] = 0.10
    return candle