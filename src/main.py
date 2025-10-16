from src.update_files import update_data, update_date
from src.backtester import backtest_strategy

def start():
    update_date.update()
    choice = input("Would you like to update the data? (y/n): ")
    if choice.lower() == 'y':
        update_data.update()
    choice = input("Would you like to backtest the strategy? (y/n): ")
    if choice.lower() == 'y':
        backtest_strategy.backtest()