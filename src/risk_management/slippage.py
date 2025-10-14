from src.utils import get_config

config = get_config.read_yaml()

def get_order_price(candle):
    for crypto in candle.key():
        candle[crypto]['order_price'] = candle[candle]['close'] * (1 + (candle[crypto]['fiducia'] * config['strategy']['slippage_cost_fraction'] / abs(candle[crypto]['fiducia'])))
    return candle