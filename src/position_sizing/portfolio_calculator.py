from src.utils import read_file

def calculate(candle):
    portfolio = read_file.read_portfolio()
    cur_holding = 0
    cryptos = list({token for (_, token) in candle.index})
    for crypto in cryptos:
        cur_holding += (portfolio.loc[crypto,'amt'] * candle[('close',crypto)])
    state = read_file.read_state()
    cur_holding += state["cash"]
    return cur_holding