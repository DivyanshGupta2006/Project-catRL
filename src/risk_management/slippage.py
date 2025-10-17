from src.utils import get_config, read_file

config = get_config.read_yaml()

def get_order_price(candle,Pt):
    portfolio = read_file.read_portfolio()
    cryptos = list({token for (_, token) in candle.index})

    for crypto in cryptos:
        a = candle[('fiducia',crypto)] * Pt / ( 1 + config['strategy']['transaction_cost_fraction'] )
        b = portfolio.loc[crypto,'amt']
        c = config['strategy']['slippage_cost_fraction']
        d = candle[('close',crypto)]

        if a / (d * (1+c)) > b:
            candle[('order_price',crypto)] = candle[('close',crypto)] * (1 + (config['strategy']['slippage_cost_fraction']))
        elif a / (d * (1-c)) < b:
            candle[('order_price',crypto)] = candle[('close',crypto)] * (1 - (config['strategy']['slippage_cost_fraction']))
        else:
            candle[('order_price',crypto)] = candle[('close',crypto)]

    return candle