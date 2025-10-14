def execute(order):
    arrange(order)

def arrange(order):
    sorted_orders = sorted(order.items(), key=lambda item: item[1]['quantity'])
    print("Processing orders: Sells first, then Buys.")
    return sorted_orders