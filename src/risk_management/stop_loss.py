from src.utils import get_config

config = get_config.read_yaml()

def get_stop_loss(candles):
    stop_loss = {}
    for symbol in candles.keys():
        stop_loss[symbol] = config['strategy']['stop_loss_multiple'] * candles['symbol']['atr']
    return stop_loss