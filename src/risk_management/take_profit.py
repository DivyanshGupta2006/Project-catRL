from src.utils import get_config

config = get_config.read_yaml()

def get_take_profit(candle):
    for crypto in candle.keys():
        candle[crypto]['take_price'] = candle[crypto]['order_price'] + config['strategy']['take_profit_multiple'] * candle[crypto]['atr']
        candle[crypto]['take_profit_amt'] = config['strategy']['take_profit_portion'] * candle[crypto]['amt']
    return candle