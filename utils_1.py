import pandas as pd
import torch
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Read and preprocess data
def load_and_preprocess_data(file_path, split_dates_ratios):
    data = pd.read_csv(file_path, parse_dates=['Date'])
    data.set_index('Date', inplace=True)
    data = data.sort_index()
    return adjust_for_splits(data, split_dates_ratios)

# Stock Split Adjustment Function
def adjust_for_splits(data, split_dates_ratios):
    for split_date, ratio in split_dates_ratios.items():
        split_date = pd.Timestamp(split_date)
        data.loc[data.index < split_date, ['Open', 'High', 'Low', 'Close']] /= ratio
        data.loc[data.index < split_date, 'Volume'] *= ratio
    return data

# Calculating Returns and Sharpe Ratio
def calculate_sharpe_ratio(data, risk_free_rate=0.03):
    data['Returns'] = data['Close'].pct_change()
    data['Sharpe_Ratio'] = (data['Returns'].rolling(window=252).mean() - risk_free_rate) / data['Returns'].rolling(window=252).std()
    data.dropna(inplace=True)
    return data

# Standardizing Data
def scale_data(data, features):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    return pd.DataFrame(scaled_data, columns=features, index=data.index), scaler

def create_dataset(tsla_data, time_step):
    x, y = [], []
    n_vectors = int(len(tsla_data) / (time_step + 1))
    for n in range(n_vectors):
        features = [tsla_data[col].values[n * (time_step + 1):n * (time_step + 1) + time_step] for col in tsla_data.columns]
        combined_features = np.column_stack(features)
        x.append(combined_features)
        y.append(tsla_data['Close'].values[n * (time_step + 1) + time_step])
    return np.array(x), np.array(y)

# Dataset segmentation
def split_dataset(X, y, split_ratio=0.8):
    split_index = int(split_ratio * len(X))
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

# Plotting the Sharpe Ratio over time
def plot_sharpe_ratio(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Sharpe_Ratio'])
    plt.title('Sharpe Ratio Over Time (Adjusted for Splits)')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    plt.show()

