from src.utils import get_config

config = get_config.read_yaml()

def get_stop_loss(candle):
    for crypto in candle.keys():
        candle[crypto]['stop_loss'] = candle[crypto]['order_price'] - config['strategy']['stop_loss_multiple'] * candle[crypto]['atr']
        candle[crypto]['stop_loss_amt'] = config['strategy']['stop_loss_portion'] * candle[crypto]['amt']
    return candle