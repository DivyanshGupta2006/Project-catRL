import ast

def convert_to_dict(row):
    row.index = row.index.map(ast.literal_eval)
    candle_df = row.unstack(level=0)
    candle = candle_df.to_dict(orient='index')
    return candle