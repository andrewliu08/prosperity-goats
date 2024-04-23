import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

round1_1 = pd.read_csv('research/round5/algo_trading/data/trades_round_1_day_-2_wn.csv', sep=';')
round1_2 = pd.read_csv('research/round5/algo_trading/data/trades_round_1_day_-1_wn.csv', sep=';')
round1_3 = pd.read_csv('research/round5/algo_trading/data/trades_round_1_day_0_wn.csv', sep=';')

round3_1 = pd.read_csv('research/round5/algo_trading/data/trades_round_3_day_0_wn.csv', sep=';')
round3_2 = pd.read_csv('research/round5/algo_trading/data/trades_round_3_day_1_wn.csv', sep=';')
round3_3 = pd.read_csv('research/round5/algo_trading/data/trades_round_3_day_2_wn.csv', sep=';')

round4_1 = pd.read_csv('research/round5/algo_trading/data/trades_round_4_day_1_wn.csv', sep=';')
round4_2 = pd.read_csv('research/round5/algo_trading/data/trades_round_4_day_2_wn.csv', sep=';')
round4_3 = pd.read_csv('research/round5/algo_trading/data/trades_round_4_day_3_wn.csv', sep=';')

SEASHELLS = "SEASHELLS"
AMETHYSTS = "AMETHYSTS"
STARFRUIT = "STARFRUIT"
ORCHIDS = "ORCHIDS"
CHOCOLATE = "CHOCOLATE"
STRAWBERRIES = "STRAWBERRIES"
ROSES = "ROSES"
GIFT_BASKET = "GIFT_BASKET"
COCONUT = "COCONUT"
COCONUT_COUPON = "COCONUT_COUPON"

r1_products = [AMETHYSTS, STARFRUIT]
r3_products = [CHOCOLATE, STRAWBERRIES, ROSES, GIFT_BASKET]
r4_products = [COCONUT, COCONUT_COUPON]

round_1 = pd.concat([round1_1, round1_2, round1_3])
round_3 = pd.concat([round3_1, round3_2, round3_3])
round_4 = pd.concat([round4_1, round4_2, round4_3])

r1_traders = round_1["buyer"].unique()
r3_traders = round_3["buyer"].unique()
r4_traders = round_4["buyer"].unique()

# print(r1_traders)
# print(r3_traders)
# print(r4_traders)
# Trader PnLs

def calc_pnl(trader, trades):
    buys = trades[trades['buyer'] == trader]['price'] * trades[trades['buyer'] == trader]['quantity']
    sells = trades[trades['seller'] == trader]['price'] * trades[trades['seller'] == trader]['quantity']
    pnl = -buys.sum() + sells.sum()
    return pnl

r1_pnls = {trader: {} for trader in r1_traders}
r3_pnls = {trader: {} for trader in r3_traders}
r4_pnls = {trader: {} for trader in r4_traders}


for trader in r1_traders:
    data = {product: round_1[round_1["symbol"] == product] for product in r1_products}
    for product, trades in data.items():
        r1_pnls[trader][product] = calc_pnl(trader, trades)

for trader in r3_traders:
    data = {product: round_3[round_3["symbol"] == product] for product in r3_products}
    for product, trades in data.items():
        r3_pnls[trader][product] = calc_pnl(trader, trades)

for trader in r4_traders:
    data = {product: round_4[round_4["symbol"] == product] for product in r4_products}
    for product, trades in data.items():
        r4_pnls[trader][product] = calc_pnl(trader, trades)

names = ["Valentina", "Vinnie", "Vladimir", "Vivian", "Celeste", "Colin", "Carlos", "Camilla", "Pablo", "Penelope", "Percy", "Petunia", "Ruby", "Remy", "Rihanna", "Raj", "Amelia", "Adam", "Alina", "Amir"]

def plot_trading_prices_detailed(trades):
    # Extract buyers and sellers
    buyers = trades['buyer'].unique()
    sellers = trades['seller'].unique()
    
    # Get all unique traders
    traders = set(buyers).union(set(sellers))
    
    # Iterate over each trader to generate separate plots for each product
    for trader in traders:
        # Filter trades involving the trader as buyer and seller, limited by timestamp
        buyer_data = trades[(trades['buyer'] == trader) & (trades['timestamp'] <= 30000)]
        seller_data = trades[(trades['seller'] == trader) & (trades['timestamp'] <= 30000)]
        
        # Get all products traded by this trader
        products = set(buyer_data['symbol']).union(set(seller_data['symbol']))
        
        for product in products:
            # Set up the plot for this trader and product
            plt.figure(figsize=(12, 8))
            
            # Filter data for this product
            product_buyer_data = buyer_data[buyer_data['symbol'] == product]
            product_seller_data = seller_data[seller_data['symbol'] == product]
            
            # Plot buying prices
            if not product_buyer_data.empty:
                plt.scatter(product_buyer_data['timestamp'], product_buyer_data['price'], label=f"{trader} Buy {product}", alpha=0.7, marker='^')
            
            # Plot selling prices
            if not product_seller_data.empty:
                plt.scatter(product_seller_data['timestamp'], product_seller_data['price'], label=f"{trader} Sell {product}", alpha=0.7, marker='v')
            
            # Customize plot with titles, labels, legend, and grid
            plt.title(f'Trading Prices Over Time for {trader} - {product}')
            plt.xlabel('Timestamp')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.show()

# Use the function to plot for different rounds
# plot_trading_prices_detailed(round_1)
# plot_trading_prices_detailed(round_3)
plot_trading_prices_detailed(round_4)