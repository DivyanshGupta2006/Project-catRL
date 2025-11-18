from src.utils import get_config, read_file, convert
from src.update_files import update_state, update_portfolio
from src.backtester import execute_SL_TP, place_order, execute_order, calculate_metrics
from src.position_sizing import portfolio_calculator, amount_calculator, fiducia_calculator
from src.risk_management import slippage, stop_loss, take_profit
from src.strategy import predict_position

class Environment:

    def __init__(self, train_data, sequence_length, num_assets):
        self.data = train_data
        self.sequence_length = sequence_length
        self.num_assets = num_assets
        self.config = get_config.read_yaml()
        self.current_step = 0

    def _assign_fiduciae(self, candle, fiduciae_action):
        # fiduciae_action is a [10,] array
        # First 9 are for cryptos, 10th is for cash
        for i, symbol in enumerate(self.crypto_symbols):
            if symbol in candle:
                # Assign the fiduciae (weight) for this crypto
                candle[symbol]['fiducia'] = fiduciae_action[i]

        return candle

    def reset(self, current_step):
        update_state.set_state(self.config['strategy']['capital'])
        update_portfolio.set_portfolio()
        self.current_step = current_step

    def step(self, fiduciae_action, buffer):

        state = read_file.read_state()
        row = self.data.loc[self.data.index[state['timestep']]].copy()
        state['timestep'] += 1
        self.timestep += 1
        update_state.update(state)

        fiduciae_action = fiducia_calculator.calculate(fiduciae_action)

        candle = convert.convert_to_dict(row)

        # not execute SL, TP here, rather execute it at the end of the last step -> simulate the next one hour after taking action
        # to get the final portfolio value : which will be used in reward

        Pt = portfolio_calculator.calculate(candle)

        candle = predict_position.assign_fiducia(candle, fiduciae_action)

        candle = slippage.get_order_price(candle, Pt)

        candle = amount_calculator.calculate(candle, Pt)

        candle = stop_loss.get_stop_loss(candle)
        candle = take_profit.get_take_profit(candle)

        order = place_order.place(candle)
        execute_order.execute(order)
        calculate_metrics.calculate_order_metrics(order)

        Pt_beg = portfolio_calculator.calculate(candle)

        done = (state['timestep'] >= len(self.data) - 1)

        if not done:
            next_row = self.data.loc[self.data.index[state['timestep']]].copy()
            next_candle = convert.convert_to_dict(next_row)
            execute_SL_TP.execute(next_candle)
            Pt_end = portfolio_calculator.calculate(next_candle)

        reward = (Pt_end - Pt_beg) / (Pt_beg + 1e-8)

        info_dict = {'reward': reward, 'profit': (Pt_end - Pt_beg)}

        buffer.states.append(self.train_data.loc[self.timestep : self.timestep + self.sequence_length - 1])
        buffer.rewards.append(reward)
        buffer.dones.append(done)

        return self.timestep, buffer

    # i look at data for t-72 -> t-1 steps, i am at end of (t - 1)'th step (72 being my sequence length for LSTM)
    # then gives action for this step, i will execute this action at the start of t'th step
    # but its reward will only be known at the end of this t'th steo - since P_new in reward is the new portfolio value - indicative of the wisdom in the model's decisions
    # this P_new is the portfolio value after executing SL, TP at end of t'th step, P_old is portfolio value after investing using these these fiducia
    # but i want to give the reward now, not wait for the next step
    # here also i update using the (t - 1)'th step only, no looking at the t'th step (to prevent look-ahead bias)
    # so for the purpose of training the model, and providing it with reward, let me execute the SL, TP for the next candle, and then calculate the P_new
