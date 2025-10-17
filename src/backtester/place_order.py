from src.utils import read_file

def place(candle):
    portfolio = read_file.read_portfolio()
    order = {}
    cryptos = list({token for (_, token) in candle.index})

    for crypto in cryptos:
        quantity = candle[('amt',crypto)] - portfolio.loc[crypto,'amt']
        if quantity != 0:
            order[crypto] = {
                'order_amt': quantity,
                'order_price': candle[('order_price',crypto)],
                'stop_price': candle[('stop_price',crypto)],
                'stop_portion': candle[('stop_portion',crypto)],
                'take_price': candle[('take_price',crypto)],
                'take_portion': candle[('take_portion',crypto)]
            }

    return order




