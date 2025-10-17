from src.utils import get_config, read_file

config = get_config.read_yaml()
trans = config['strategy']['transaction_cost_fraction']

def calculate(candle, Pt):
    cryptos = list({token for (_, token) in candle.index})

    for crypto in cryptos:
        if candle[('order_price',crypto)] != candle[('close',crypto)]:
            candle[('amt', crypto)] = ((candle[('fiducia',crypto)] * Pt) / (candle[('order_price',crypto)] * (1 + trans)))
        else:
            portfolio = read_file.read_portfolio()
            candle[('amt',crypto)] = portfolio.loc[crypto, 'amt']
    return candle