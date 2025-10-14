from src.utils import get_config

config = get_config.read_yaml()

def get_slippage(candle):
    for crypto in candle.key():
        candle[crypto]['order_price'] = candle[candle]['close'] * (1 + (candle[crypto]['fiducias'] * config['strategy']['slippage_cost_fraction'] / abs(candle[crypto]['fiducias'])))
    return candle