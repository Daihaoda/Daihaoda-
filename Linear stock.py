import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import os
pd.set_option('expand_frame_repr', False)  # 当列太多时清楚展示
df2 = pd.read_csv(
    r'C:\Users\戴浩达\Desktop\文件\量化\Earning\TSLA.csv',
    encoding='gbk',
    # index_col=['日期']
)
df1 = pd.read_csv(
    r'C:\Users\戴浩达\Desktop\文件\量化\Earning\SPY.csv',
    encoding='gbk',
    # index_col=['Date']
)


df2['Date'] = pd.to_datetime(df2['Date'])
df1['Date'] = pd.to_datetime(df1['Date'])


common_dates = df1.index.intersection(df2.index)


df1_common = df1.loc[common_dates]
df2_common = df2.loc[common_dates]


df_merged = pd.merge(df1_common, df2_common, on='Date', suffixes=('_stock1', '_stock2'))


df_merged['log_return_stock1'] = np.log(df_merged['Close_stock1'] / df_merged['Close_stock1'].shift(1))
df_merged['log_return_stock2'] = np.log(df_merged['Close_stock2'] / df_merged['Close_stock2'].shift(1))

print(df_merged)


df_merged.dropna(inplace=True)


X = df_merged['log_return_stock1']
y = df_merged['log_return_stock2']
X = sm.add_constant(X)

model = sm.OLS(y, X)
results = model.fit()


print(results.summary())
# Assuming df_merged is your DataFrame after merging and cleaning
window_size = 252  # Approximate trading days in a year


def rolling_regression(y, x, window):
    """
    Perform rolling regression of y on x.

    :param y: Dependent variable series.
    :param x: Independent variable series.
    :param window: Size of the rolling window.
    :return: Series of regression coefficients.
    """

    def regression(y_window):
        if len(y_window.dropna()) == window:
            # Use .loc to ensure label-based indexing
            x_window = x.loc[y_window.index]
            # Add a constant for the regression
            x_with_const = sm.add_constant(x_window)
            # Fit the model
            model = sm.OLS(y_window, x_with_const).fit()
            # Return the coefficient (slope) of the independent variable
            return model.params[1]
        else:
            return np.nan

    # Apply the regression function over a rolling window
    coeffs = y.rolling(window=window).apply(regression, raw=False)

    return coeffs


# Step 2: Calculate the rolling regression coefficients
window_size = 252  # Approximate trading days in a year
df_merged['beta'] = rolling_regression(df_merged['log_return_stock2'], df_merged['log_return_stock1'], window_size)

# Step 3: Plot the coefficients
plt.figure(figsize=(14, 7))
plt.plot(df_merged['Date'], df_merged['beta'], label='Rolling Beta')
plt.title('Rolling Regression Coefficient Over Time')
plt.xlabel('Date')
plt.ylabel('Beta')
plt.legend()
plt.show()