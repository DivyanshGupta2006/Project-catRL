import torch
import matplotlib.pyplot as plt

from src.backtester import place_order, execute_order, calculate_metrics, execute_SL_TP
from src.position_sizing import fiducia_calculator, portfolio_calculator, amount_calculator
from src.update_files import update_state, update_portfolio
from src.risk_management import slippage, stop_loss, take_profit
from src.utils import convert

class Environment:

    def __init__(self,
                 data,
                 bound_reward_factor,
                 seq_len,
                 capital,
                 symbols,
                 results_path):
        self.data = data
        self.bound_reward_factor = bound_reward_factor
        self.seq_len = seq_len
        self.capital = capital
        self.symbols = symbols
        self.results_path = results_path

        for idx, symbol in enumerate(self.symbols):
            self.symbols[idx] = symbol.split('/')[0]

        self.current_step = 10 * self.seq_len
        self.prev_portfolio = self.capital
        self.equity = []
        self.reset_counter = 0

    def _get_states(self, field_of_view):
        try:
            obs = field_of_view.iloc[self.current_step - self.seq_len : self.current_step].copy()
            states = torch.tensor(obs.values, dtype=torch.float32)
            return states.unsqueeze(0)
        except:
            return None

    def _get_reward(self, prev, new, flag, fiduciae):
        bound_reward = 0
        for bound in flag:
            if bound == 'sl':
                bound_reward -= 1
            elif bound == 'tp':
                bound_reward += 1

        bound_reward = bound_reward * self.bound_reward_factor * prev

        reward = (new + bound_reward - prev) / prev

        for fiducia in fiduciae:
            if abs(fiducia) <= 1e-7:
                reward = reward - 1

        return reward

    def step(self, raw_action, field_of_view):
        row1 = self.data.iloc[self.current_step - 1]
        row2 = self.data.iloc[self.current_step]
        prev_candle = convert.convert_to_dict(row1)
        candle = convert.convert_to_dict(row2)

        fiduciae = fiducia_calculator.calculate(raw_action)

        fiduciae = fiduciae.tolist()

        for idx, crypto in enumerate(self.symbols):
            prev_candle[crypto]['fiducia'] = fiduciae[idx]

        prev_candle = slippage.get_order_price(prev_candle, self.prev_portfolio)
        prev_candle = amount_calculator.calculate(prev_candle, self.prev_portfolio)
        prev_candle = stop_loss.get_stop_loss(prev_candle)
        prev_candle = take_profit.get_take_profit(prev_candle)

        order = place_order.place(prev_candle)
        execute_order.execute(order)
        calculate_metrics.calculate_order_metrics(order)
        flag = execute_SL_TP.execute(candle)
        calculate_metrics.calculate_candle_metrics(candle)

        new_portfolio = portfolio_calculator.calculate(candle)
        self.equity.append(new_portfolio)
        if(new_portfolio < 0.001 * self.capital):
            done = 1
        else:
            done = 0

        if done == 0:
            reward = self._get_reward(self.prev_portfolio, new_portfolio, flag, fiduciae)
        else:
            reward = -9

        # reward = self._get_reward(self.prev_portfolio, new_portfolio, flag, fiduciae)

        self.current_step += 1
        next_states = self._get_states(field_of_view)

        return next_states, reward, done

    def reset(self, field_of_view, to_plot=False):
        update_state.set_state(self.capital)
        update_portfolio.set_portfolio()
        # self.current_step = self.seq_len
        self.prev_portfolio = self.capital
        if to_plot:
            plt.plot(self.equity)
            plt.title('Equity Curve')
            plt.ylabel('Equity')
            plt.xlabel('Time')
            # plt.show()
            plt.savefig(self.results_path / f'equity_curve{self.reset_counter}.png', dpi=300, bbox_inches='tight')
            plt.close()
        self.equity = []
        self.reset_counter += 1
        return self._get_states(field_of_view)