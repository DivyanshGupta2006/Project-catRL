from src.utils import get_config, read_file

config = get_config.read_yaml()

def get_stop_loss(candle):
    cryptos = list({token for (_, token) in candle.index})
    for crypto in cryptos:
        candle[('stop_price',crypto)] = candle[('order_price',crypto)] + ( config['strategy']['stop_loss_multiple'] * candle[('atr',crypto)] * candle[('fiducia',crypto)] / abs(candle[('fiducia',crypto)]) )
        candle[('stop_portion',crypto)] = config['strategy']['stop_loss_portion'] * candle[('amt',crypto)]
    return candle