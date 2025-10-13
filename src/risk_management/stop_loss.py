from src.utils import get_config

config = get_config.read_yaml()

def get_stop_loss(candles):
    stop_loss = {}
    for symbol in candles.keys():
        candles[symbol]['stop_loss'] = candles[symbol]['order_price'] - config['strategy']['stop_loss_multiple'] * candles[symbol]['atr']
        candles[symbol]['stop_loss_amt'] = config['strategy']['stop_loss_portion'] * candles[symbol]['amt']
    return stop_loss