from src.backtester import data_handler, place_order, execute_order, rebalance_portfolio
from src.agent import predict
from src.risk_management import slippage, stop_loss, take_profit
from src.position_sizing import amount_calculator

def backtest_strategy(data):
    candle = data_handler.fetch_data(data)
    candle = predict.predict_position(candle)
    candle = slippage.get_order_price(candle)
    candle = amount_calculator.calculate(candle)
    candle = stop_loss.get_stop_loss(candle)
    candle = take_profit.get_take_profit(candle)
    order = place_order.place(candle)
    execute_order.execute(order)
    rebalance_portfolio.rebalance(order)