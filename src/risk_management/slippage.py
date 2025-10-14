from src.utils import get_config

config = get_config.read_yaml()

def get_slippage(candle):
    for candle in candle.key():
        candle[candle]['order_price'] = candle[candle]['close'] * config['strategy']['slippage_cost_fraction']
    return candle