from src.utils import get_config, read_file

config = get_config.read_yaml()

def get_order_price(candle,Pt):
    portfolio = read_file.read_portfolio()

    for crypto in candle.key():
        a = candle[crypto]['fiducia'] * Pt / ( 1 + config['strategy']['transaction_cost_fraction'] )
        b = portfolio.loc[crypto,'amt']
        c = config['strategy']['slippage_cost_fraction']
        d = candle[crypto]['close']

        if a / (d * (1+c)) > b:
            candle[crypto]['order_price'] = candle[crypto]['close'] * (1 + (config['strategy']['slippage_cost_fraction']))
        elif a / (d * (1-c)) < b:
            candle[crypto]['order_price'] = candle[crypto]['close'] * (1 - (config['strategy']['slippage_cost_fraction']))
        else:
            candle[crypto]['order_price'] = candle[crypto]['close']

    return candle