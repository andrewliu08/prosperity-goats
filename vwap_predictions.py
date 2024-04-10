
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score


price_2 = pd.read_csv('r1_data/prices_round_1_day_-2.csv', sep=';')
price_1 = pd.read_csv('r1_data/prices_round_1_day_-1.csv', sep=';')
price_0 = pd.read_csv('r1_data/prices_round_1_day_0.csv', sep=';')

trades_2 = pd.read_csv('r1_data/trades_round_1_day_-2_nn.csv', sep=';')
trades_1 = pd.read_csv('r1_data/trades_round_1_day_-1_nn.csv', sep=';')
trades_0 = pd.read_csv('r1_data/trades_round_1_day_0_nn.csv', sep=';')

all_prices = pd.concat([price_2, price_1, price_0])
all_trades = pd.concat([trades_2, trades_1, trades_0])

prices = {"all": all_prices}
trades = {"all": all_trades}
products = list(all_prices['product'].unique())
for product in products:
    prices[product] = all_prices[all_prices['product'] == product]
    trades[product] = all_trades[all_trades['symbol'] == product]

# PLOT DISTRIBUTIONS
price_columns = ["mid_price", "bid_price_1", "bid_price_2", "bid_price_3", "ask_price_1", "ask_price_2", "ask_price_3"]

# CONFIGS
product = "STARFRUIT"
# scaler = MinMaxScaler()
model = Ridge()

# SCALE DATA
df = prices[product].copy()
df.reset_index(drop=True, inplace=True)
# df_scaled = pd.DataFrame(scaler.fit_transform(prices[product][price_columns]), columns=price_columns)
# for column in price_columns:
#     df[column] = df_scaled[column]

# df.head()
# FEATURE EXTRACTION

features = pd.DataFrame()

total_bid_volume = (
    df['bid_volume_1'].fillna(0) +
    df['bid_volume_2'].fillna(0) +
    df['bid_volume_3'].fillna(0)
)
total_ask_volume = (
    df['ask_volume_1'].fillna(0) +
    df['ask_volume_2'].fillna(0) +
    df['ask_volume_3'].fillna(0)
)


# Weighted Price: Weighted price for bids and asks
features['weighted_bid_price'] = (
    df['bid_price_1'].fillna(0) * df['bid_volume_1'].fillna(0) +
    df['bid_price_2'].fillna(0) * df['bid_volume_2'].fillna(0) +
    df['bid_price_3'].fillna(0) * df['bid_volume_3'].fillna(0)
) / total_bid_volume

features['weighted_ask_price'] = (
    df['ask_price_1'].fillna(0) * df['ask_volume_1'].fillna(0) +
    df['ask_price_2'].fillna(0) * df['ask_volume_2'].fillna(0) +
    df['ask_price_3'].fillna(0) * df['ask_volume_3'].fillna(0)
) / total_ask_volume

# Weighted Price: Weighted price for bids and asks
features['weighted_trade_price'] = (
    df['bid_price_1'].fillna(0) * df['bid_volume_1'].fillna(0) +
    df['bid_price_2'].fillna(0) * df['bid_volume_2'].fillna(0) +
    df['bid_price_3'].fillna(0) * df['bid_volume_3'].fillna(0) +
    df['ask_price_1'].fillna(0) * df['ask_volume_1'].fillna(0) +
    df['ask_price_2'].fillna(0) * df['ask_volume_2'].fillna(0) +
    df['ask_price_3'].fillna(0) * df['ask_volume_3'].fillna(0)
) / (total_bid_volume + total_ask_volume)

previous_timesteps = 5
future_timesteps = 1

# feature calc

# Previous Timesteps:
for lag in range(0, previous_timesteps):
    features[f'lag_{lag}'] = features['weighted_trade_price'].shift(lag)

features['ema_weighted_price'] = features['weighted_trade_price'].ewm(span=10, adjust=False).mean()
# features['vwap_midprice_product'] = features['weighted_trade_price'] * df['mid_price'].reset_index(drop=True)

features['ma5'] = features['weighted_trade_price'].rolling(window=5).mean()
features['ma20'] = features['weighted_trade_price'].rolling(window=20).mean()

# RSI Calculation
delta = df['mid_price'].diff(1)
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
features['rsi'] = 100 - (100 / (1 + rs))

features['log_returns'] = np.log(features['weighted_trade_price'] / features['weighted_trade_price'].shift(1))
# Handling any infinite or NaN values that might arise from division by zero or log of zero
features['log_returns'] = features['log_returns'].replace([np.inf, -np.inf], np.nan).fillna(0)

exp1 = df['mid_price'].ewm(span=12, adjust=False).mean()
exp2 = df['mid_price'].ewm(span=26, adjust=False).mean()
features['macd'] = exp1 - exp2
features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()

# featurecalc

# Future Timesteps:
features['future'] = features['weighted_trade_price'].shift(-future_timesteps)
features = features.dropna()

# print(features.tail())
selected_columns=['log_returns', 'ma5', 'ma20','weighted_bid_price','weighted_ask_price']
X = features[selected_columns]
# X = features.drop(columns=['future'])
y = features['future']

# TRAIN-TEST SPLIT
split_point = int(len(X) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]


# PREDICTIONS
model.fit(X_train, y_train)
predictions = model.predict(X_test)


# vwap_values = features['weighted_trade_price']

# # Calculate the average and standard deviation of VWAP
# average_vwap = vwap_values.mean()
# std_dev_vwap = vwap_values.std()

# print(f"Average VWAP: {average_vwap}")
# print(f"Standard Deviation of VWAP: {std_dev_vwap}")
# ERROR

mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test, predictions)
print("R2 Score:", r2)

mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)

mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
print("Mean Absolute Percentage Error:", mape, "%")

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("Root Mean Squared Error:", rmse)

explained_variance = explained_variance_score(y_test, predictions)
print("Explained Variance Score:", explained_variance)

# COEFFICIENTS
def plot_coefficients(model, feature_names):
    # Get coefficients and feature names
    coefficients = model.coef_
    names = feature_names
    # Sort coefficients and names by absolute value
    sorted_indices = np.argsort(np.abs(coefficients))[::-1]
    sorted_coefficients = coefficients[sorted_indices]
    sorted_names = [names[i] for i in sorted_indices]
    # Plot coefficient importance
    # plt.figure(figsize=(10, 6))
    # plt.barh(range(len(sorted_names)), sorted_coefficients, align='center')
    # plt.yticks(range(len(sorted_names)), sorted_names)
    # plt.xlabel('Coefficient Value')
    # plt.title('Coefficient Importance')
    # plt.show()

    for i, (coef, column) in enumerate(zip(sorted_coefficients, sorted_names)):
        print(f"{column.ljust(20)}: {coef}")
    
    # Plot Actual vs. Predicted Prices
    plt.figure(figsize=(10, 6))
    y_test_subset = y_test.reset_index(drop=True)[:200]
    predictions_subset = predictions[:200]
    plt.plot(y_test_subset, label='Actual')
    plt.plot(predictions_subset, label='Predicted', alpha=0.7)
    plt.title('Actual vs. Predicted Prices')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Manually calculate the Mean Squared Error (MSE) for the first 50 predictions
    mse_manual = np.sum((y_test_subset - predictions_subset) ** 2) / len(y_test_subset)
    print(f"Manually Calculated Mean Squared Error: {mse_manual}")

    weighted_trade_price_diff = features['weighted_trade_price'].diff().abs().dropna()
    
    average_diff = weighted_trade_price_diff.mean()
    print(f"Average Difference of the Weighted Trade Price (VWAP) for Consecutive Entries: {average_diff}")

# Ensure to pass the actual y_test and predictions arrays to this function when calling it
plot_coefficients(model, X.columns)