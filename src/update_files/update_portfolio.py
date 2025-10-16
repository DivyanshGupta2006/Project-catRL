from src.utils import get_config, get_absolute_path

config = get_config.read_yaml()

def update(portfolio):
    dir = get_absolute_path.absolute(config['paths']['portfolio_directory'])
    portfolio.to_csv(dir)