from src.utils import read_file

def calculate(candle):
    portfolio = read_file.read_portfolio()
    cur_holding = 0
    for crypto in candle:
        cur_holding += (portfolio.loc[crypto,'amt'] * candle[crypto]['close'])
    state = read_file.read_state()
    cur_holding += state["cash"]
    return cur_holding