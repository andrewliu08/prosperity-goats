import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
import math

prices_2 = pd.read_csv('research/round4/algo_trading/data/prices_round_4_day_3.csv', sep=';')
prices_1 = pd.read_csv('research/round4/algo_trading/data/prices_round_4_day_2.csv', sep=';')
prices_0 = pd.read_csv('research/round4/algo_trading/data/prices_round_4_day_1.csv', sep=';')

trades_2 = pd.read_csv('research/round4/algo_trading/data/trades_round_4_day_3_nn.csv', sep=';')
trades_1 = pd.read_csv('research/round4/algo_trading/data/trades_round_4_day_2_nn.csv', sep=';')
trades_0 = pd.read_csv('research/round4/algo_trading/data/trades_round_4_day_1_nn.csv', sep=';')

df = pd.concat([prices_0]).reset_index(drop=True)

df.head()

df["product"].unique()

coconut_prices = df[df["product"] == "COCONUT"].reset_index(drop=True)
coupon_prices = df[df["product"] == "COCONUT_COUPON"].reset_index(drop=True)

cols = ["day", "timestamp", "mid_price"]
merge = pd.merge(coconut_prices[cols], coupon_prices[cols], on=['day', 'timestamp'], how='inner', suffixes=('_coconut', '_coupon'))

merge.rename(columns={"mid_price": "mid_price_coconut",}, inplace=True)
merge.head()

merge['shifted_coconut_price'] = merge['mid_price_coconut'].shift(-1)
merge = merge.dropna()
merge['unit_increase_coconut'] = merge['mid_price_coconut'] - merge['shifted_coconut_price']
merge['unit_increase_coconut'].describe()

sorted = merge.sort_values(by='unit_increase_coconut', ascending=True)

merge = merge[merge['unit_increase_coconut'] > -9]
value_counts = merge['unit_increase_coconut'].value_counts()
value_counts_filtered = value_counts[value_counts.index != -253.0]
## CDF??
probability_distribution = value_counts_filtered / value_counts_filtered.sum()
custom_cdf_data = probability_distribution.sort_index().cumsum()

x = custom_cdf_data.index.values
y = custom_cdf_data.values
custom_cdf = interp1d(x, y, bounds_error=False, fill_value=(y.min(), y.max()))
merge['log_price_coconut'] = np.log(merge['mid_price_coconut'])
merge['shifted_log_price_coconut'] = merge['log_price_coconut'].shift(-1)
merge['log_returns_coconut'] = merge['log_price_coconut'] - merge['shifted_log_price_coconut']
merge = merge.dropna()
merge['log_returns_coconut'].describe()

coconut_prices['expanding_volatility'] = merge['log_returns_coconut'].expanding(min_periods=1).std() * np.sqrt(252) * 100

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        option_price = (S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2))
    else:
        option_price = (K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1))
    
    return option_price

S = coconut_prices['mid_price']  # Current coconut price (spot price)
K = 10000  # Strike price
r = 0  # Risk-free rate
T = 250/252 # Time to maturity
volatility = 0.1605

coconut_prices['option_price'] = coconut_prices.apply(
    lambda row: black_scholes(row['mid_price'], K, T, r, volatility), axis=1)

first_option_price = coconut_prices['option_price'].iloc[0]
print("The first option price is:", first_option_price)
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(coconut_prices['timestamp'], coconut_prices['option_price'], label='Option Price')
plt.title('Daily Option Price for Coconut Coupon')
plt.xlabel('Timestamp')
plt.ylabel('Option Price ($)')
plt.grid(True)
plt.legend()
plt.show()

# Convert 'timestamp' to datetime if not already done
coconut_prices['timestamp'] = pd.to_datetime(coconut_prices['timestamp'])
coupon_prices['timestamp'] = pd.to_datetime(coupon_prices['timestamp'])

# Merge the DataFrames on 'timestamp'
comparison_df = pd.merge(coconut_prices[['timestamp', 'option_price']],
                         coupon_prices[['timestamp', 'mid_price']],
                         on='timestamp',
                         how='inner',
                         suffixes=('_option', '_coupon'))

# Calculate R^2 score
r_squared = r2_score(comparison_df['mid_price'], comparison_df['option_price'])
print(f"R^2 score between option price and coupon mid price: {r_squared:.3f}")

# Plotting both prices over time
plt.figure(figsize=(14, 7))
plt.plot(comparison_df['timestamp'], comparison_df['option_price'], label='Coconut Option Price', linestyle='-')
plt.plot(comparison_df['timestamp'], comparison_df['mid_price'], label='Coupon Mid Price', linestyle='--')

# Setting the title and labels
plt.title('Option Price vs Coupon Mid Price Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Price')
plt.xticks(rotation=45)  # Rotating the timestamps for better visibility
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# have call option price
# need CDF (computed from the log returns, they only vary from -4 to 4 in differences of 0.5, do this manually)
# have spot price as coconut price
# have strike price as 10,000
# sub r=standard interest
# sub t=250/365
# need volatility (stddev of log returns)
