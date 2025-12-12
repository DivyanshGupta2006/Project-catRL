from src.update_files import update_date, update_dir

def start():
    update_date.update()
    update_dir.update()
    choice = input("Would you like to update the data? (y/n): ")
    if choice.lower() == 'y':
        from src.update_files import update_data
        update_data.update()
    choice = input("Would you like to train the agent? (y/n): ")
    if choice.lower() == 'y':
        from src.strategy import train_agent
        from src.backtester import backtest_strategy
        train_agent.train()
        print('Backtesting on validation set...')
        backtest_strategy.backtest_on_val()
        print('Backtesting on test set...')
        backtest_strategy.backtest_on_test()

    else:

        choice = input("Would you like to backtest the strategy on validation set? (y/n): ")
        if choice.lower() == 'y':
            from src.backtester import backtest_strategy
            backtest_strategy.backtest_on_val()

        choice = input("Would you like to backtest the strategy on test set? (y/n): ")
        if choice.lower() == 'y':
            from src.backtester import backtest_strategy
            backtest_strategy.backtest_on_test()

        choice = input("Would you like to backtest the strategy on train set? (y/n): ")
        if choice.lower() == 'y':
            from src.backtester import backtest_strategy
            backtest_strategy.backtest_on_train()
