from src.utils import get_config

config = get_config.read_yaml()

def get_take_profit(candles):
    for symbol in candles.keys():
        candles[symbol]['take_price'] = candles[symbol]['order_price'] + config['strategy']['take_profit_multiple'] * candles[symbol]['atr']
        candles[symbol]['take_profit_amt'] = config['strategy']['take_profit_portion'] * candles[symbol]['amt']
    return candles