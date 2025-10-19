from src.utils import get_config, read_file

config = get_config.read_yaml()

def get_order_price(candle,Pt):
    portfolio = read_file.read_portfolio()

    for crypto in candle:
        a = candle[crypto]['close']
        S = config['strategy']['slippage_cost_fraction']
        K = candle[crypto]['fiducia'] * Pt / (1 + config['strategy']['transaction_cost_fraction'])
        y0 = portfolio.loc[crypto,'amt']

        exp1 = K / (a * (1 + S))
        exp2 = K / (a * (1 - S))

        if exp1 > y0:
            candle[crypto]['order_price'] = a * (1 + S)
        elif exp2 < y0:
            candle[crypto]['order_price'] = a * (1 - S)
        else:
            candle[crypto]['order_price'] = portfolio.loc[crypto,'order_price']

    return candle