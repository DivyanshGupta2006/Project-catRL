def place(candle, portfolio):
    order = {}

    for crypto in candle:
        quantity = candle[crypto]['amt'] - portfolio.loc[crypto,'amt']
        order[crypto] = {
            'order_amt': quantity,
            'order_price': candle[crypto]['order_price'],
            'stop_price': candle[crypto]['stop_price'],
            'stop_portion': candle[crypto]['stop_portion'],
            'take_price': candle[crypto]['take_price'],
            'take_portion': candle[crypto]['take_portion']
        }

    return order




