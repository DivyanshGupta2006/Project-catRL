from src.utils import get_config

config = get_config.read_yaml()

def get_stop_loss(candle):
    for crypto in candle:
        if candle[crypto]['fiducia'] != 0:
            candle[crypto]['stop_price'] = candle[crypto]['order_price'] - ( config['strategy']['stop_loss_multiple'] * candle[crypto]['atr'] * candle[crypto]['fiducia'] / abs(candle[crypto]['fiducia']) )
            candle[crypto]['stop_portion'] = config['strategy']['stop_loss_portion'] * candle[crypto]['amt']
        else:
            candle[crypto]['stop_price'] = 0
            candle[crypto]['stop_portion'] = 0
    return candle