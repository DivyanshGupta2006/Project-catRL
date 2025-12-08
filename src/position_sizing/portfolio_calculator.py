def calculate(candle, state, portfolio):
    cur_holding = 0
    for crypto in candle:
        cur_holding += (portfolio.loc[crypto,'amt'] * candle[crypto]['close'])
    cur_holding += state["cash"]
    return cur_holding