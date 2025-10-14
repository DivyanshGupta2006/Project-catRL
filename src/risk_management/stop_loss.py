from src.utils import get_config

config = get_config.read_yaml()

def get_stop_loss(candle):
    for symbol in candle.keys():
        candle[symbol]['stop_loss'] = candle[symbol]['order_price'] - config['strategy']['stop_loss_multiple'] * candle[symbol]['atr']
        candle[symbol]['stop_loss_amt'] = config['strategy']['stop_loss_portion'] * candle[symbol]['amt']
    return candle