from src.utils import get_config

config = get_config.read_yaml()

def get_slippage(candles):
    for candle in candles.key():
        candles[candle]['order_price'] = candles[candle]['close'] * config['strategy']['slippage_fraction']
    return candles