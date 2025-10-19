from src.risk_management import slippage
from src.utils import get_config, read_file

config = get_config.read_yaml()
trans = config['strategy']['transaction_cost_fraction']

def calculate(candle, Pt):

    portfolio = read_file.read_portfolio()

    for crypto in candle:
        if candle[crypto]['order_price'] == portfolio.loc[crypto, 'order_price']:
            candle[crypto]['amt'] = portfolio.loc[crypto, 'amt']
        else:
            K = candle[crypto]['fiducia'] * Pt / (1 + config['strategy']['transaction_cost_fraction'])
            candle[crypto]['amt'] = K / candle[crypto]['order_price']
    return candle