from src.utils import get_config
from src.position_sizing import portfolio_calculator

config = get_config.read_yaml()
symbols = [symbol.strip('/')[0] for symbol in config['data']['symbols']]
slip = config['strategy']['slippage_cost_fraction']
trans = config['strategy']['transaction_cost_fraction']

def calculate_amount(candle):
    portfolio_value = portfolio_calculator.calc_portfolio()
    for crypto in candle.keys():
        candle[crypto]['amt'] = round((candle[crypto]['fiducias'] * portfolio_value) / (candle[crypto]['close'] * (1 + slip) * (1 + trans)),4)
    return candle

# test = {
#     'ETH' : {'close' : 20, 'fiducias' : 10},
#     'BTC' : {'close' : 30, 'fiducias' : 40},
# }
#
# print(calculate_amount(test))