from src.position_sizing import portfolio_calculator
from src.utils import read_file, get_config

config = get_config.read_yaml()

def print_results(candle, data):
    state = read_file.read_state()
    portfolio = read_file.read_portfolio()
    print('--------------Backtesting Results---------------')
    print(f'Initial portfolio value: {config['strategy']['capital']} USDT')
    print(f'Start date: {data.index[0]}')
    print(f'End date: {state["timestep"]}')
    print('--------------Final Portfolio--------------')
    print(portfolio)
    print('-------------------------------------------')
    print(f'Final portfolio value: {portfolio_calculator.calculate(candle)} USDT')
    print(f'Final cash: {state["cash"]} USDT')
    print(f'Returns: {state['metrics']["returns"] * 100} %')
    print(f'Peak Equity: {state['metrics']["peak equity"]} USDT')
    print(f'Drawdown: {state['metrics']["drawdown"] * 100} %')
    print(f'Total transaction cost: {state['metrics']["transaction_cost"]} USDT')
    print('------------------------------------------------')