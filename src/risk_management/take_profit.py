from src.utils import get_config, read_file

config = get_config.read_yaml()

def get_take_profit(candle):
    cryptos = list({token for (_, token) in candle.index})
    for crypto in cryptos:
        candle[('take_price',crypto)] = candle[('order_price',crypto)] + ( config['strategy']['take_profit_multiple'] * candle[('atr',crypto)] * candle[('fiducia',crypto)] / abs(candle[('fiducia',crypto)]) )
        candle[('take_portion',crypto)] = config['strategy']['take_profit_portion'] * candle[('amt',crypto)]
    return candle