from src.utils import get_config

config = get_config.read_yaml()

def predict(candle):
    for crypto in candle:
        # assign different if want to test different fiducia
        if crypto == 'ETH' or crypto == 'BTC':
            candle[crypto]['fiducia'] = 0.10
        else:
            candle[crypto]['fiducia'] = 0.10
    return candle