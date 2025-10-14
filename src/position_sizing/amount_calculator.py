from src.utils import get_config
from src.position_sizing import portfolio_calculator

config = get_config.read_yaml()
symbols = [symbol.strip('/')[0] for symbol in config['data']['symbols']]
trans = config['strategy']['transaction_cost_fraction']

def calculate(candle):
    portfolio_value = portfolio_calculator.calculate()
    for crypto in candle.keys():
        candle[crypto]['amt'] = ((candle[crypto]['fiducia'] * portfolio_value) / (candle[crypto]['order_price'] * (1 + trans)))
    return candle