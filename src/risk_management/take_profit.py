from src.utils import get_config

config = get_config.read_yaml()

def get_take_profit(candles):
    take_profit = {}
    for symbol in candles.keys():
        take_profit[symbol] = config['strategy']['take_profit_multiple'] * candles['symbol']['atr']
    return take_profit