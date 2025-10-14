from src.utils import get_config

config = get_config.read_yaml()

def get_take_profit(candle):
    for symbol in candle.keys():
        candle[symbol]['take_price'] = candle[symbol]['order_price'] + config['strategy']['take_profit_multiple'] * candle[symbol]['atr']
        candle[symbol]['take_profit_amt'] = config['strategy']['take_profit_portion'] * candle[symbol]['amt']
    return candle