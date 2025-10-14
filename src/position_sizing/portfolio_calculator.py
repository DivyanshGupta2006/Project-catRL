from src.utils import read_file

def calculate():
    portfolio = read_file.read_portfolio()
    current_values = portfolio['amt'] * portfolio['order_price']
    cur_holding = current_values.sum()
    state = read_file.read_state()
    cur_holding += state["cash"]
    return cur_holding