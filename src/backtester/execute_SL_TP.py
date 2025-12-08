from src.utils import get_config

config = get_config.read_yaml()

def execute(candle, state, portfolio):
    ret = []

    for crypto in portfolio.index:
        # shorting
        if portfolio.loc[crypto, 'amt'] < 0:
            if candle[crypto]['high'] >= portfolio.loc[crypto, 'stop_price']:
                portfolio.loc[crypto, 'amt'] -= portfolio.loc[crypto, 'stop_portion']
                state['cash'] += (portfolio.loc[crypto, 'stop_portion'] * portfolio.loc[crypto, 'stop_price'])
                ret.append('sl')

            elif candle[crypto]['low'] <= portfolio.loc[crypto, 'take_price']:
                portfolio.loc[crypto, 'amt'] -= portfolio.loc[crypto, 'take_portion']
                state['cash'] += (portfolio.loc[crypto, 'take_portion'] * portfolio.loc[crypto, 'take_price'])
                ret.append('tp')

            else:
                ret.append('na')

        # longing
        elif portfolio.loc[crypto, 'amt'] > 0:
            if candle[crypto]['low'] <= portfolio.loc[crypto, 'stop_price']:
                portfolio.loc[crypto, 'amt'] -= portfolio.loc[crypto, 'stop_portion']
                state['cash'] += (portfolio.loc[crypto, 'stop_portion'] * portfolio.loc[crypto, 'stop_price'])
                ret.append('sl')

            elif candle[crypto]['high'] >= portfolio.loc[crypto, 'take_price']:
                portfolio.loc[crypto, 'amt'] -= portfolio.loc[crypto, 'take_portion']
                state['cash'] += (portfolio.loc[crypto, 'take_portion'] * portfolio.loc[crypto, 'take_price'])
                ret.append('tp')

            else:
                ret.append('na')

    return ret, state, portfolio