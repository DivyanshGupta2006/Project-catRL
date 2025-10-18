from src.utils import get_config, get_absolute_path, read_file

config = get_config.read_yaml()

def update(portfolio):
    dir = get_absolute_path.absolute(config['paths']['portfolio_directory'])
    portfolio.to_csv(dir)

def set_portfolio():
    symbols = config['data']['symbols']
    new_symbols = []
    for symbol in symbols:
        symbol = symbol.split('/')[0]
        new_symbols.append(symbol)
    portfolio = read_file.read_portfolio()
    portfolio['symbol'] = new_symbols
    portfolio[['amt', 'order_price', 'stop_price', 'stop_portion', 'take_price', 'take_portion']] = 0
    dir = get_absolute_path.absolute(config['paths']['portfolio_directory'])
    portfolio.to_csv(dir, index=False)