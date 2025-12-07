from src.update_files import update_date

def start():
    update_date.update()
    choice = input("Would you like to update the data? (y/n): ")
    if choice.lower() == 'y':
        from src.update_files import update_data
        update_data.update()
    choice = input("Would you like to train the agent? (y/n): ")
    if choice.lower() == 'y':
        from src.strategy import train_agent
        train_agent.train()
    choice = input("Would you like to backtest the strategy? (y/n): ")
    if choice.lower() == 'y':
        from src.backtester import backtest_strategy
        backtest_strategy.backtest()
