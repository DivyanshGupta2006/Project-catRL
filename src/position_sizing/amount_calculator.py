from src.utils import get_config, read_file

config = get_config.read_yaml()
trans = config['strategy']['transaction_cost_fraction']

def calculate(candle, Pt):

    for crypto in candle.keys():
        if candle[crypto]['order_price'] != candle[crypto]['close']:
            candle[crypto]['amt'] = ((candle[crypto]['fiducia'] * Pt) / (candle[crypto]['order_price'] * (1 + trans)))
        else:
            portfolio = read_file.read_portfolio()
            candle[crypto]['amt'] = portfolio.loc[crypto, 'amt']
    return candle