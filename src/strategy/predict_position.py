from src.utils import get_config

config = get_config.read_yaml()

symbols = config['data']['symbols']
symbols = [symbol.split('/')[0] for symbol in symbols]

def predict_fiduciae(candle):
    return [0.22, 0.22, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.0]

def assign_fiducia(candle, fiduciae):
    for idx,crypto in enumerate(symbols):
        candle[crypto]['fiducia'] = fiduciae[idx]

    return candle

def predict(candle):
    fiduciae = predict_fiduciae(candle)
    candle = assign_fiducia(candle, fiduciae)
    return candle