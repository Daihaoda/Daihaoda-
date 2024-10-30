# 📈 Stock Price Prediction using Time-Series Models

![Project Banner](image/Project_Banner.png) <!-- 可选，增加视觉效果 -->

---

## 📋 Project Overview

在这个项目中，我们探索如何使用不同的**时序预测模型**来预测股票价格。主要内容包括数据预处理、特征工程、模型训练和预测评估。

---

## 🔧 Technologies and Tools

- **Python**
- **Pandas**, **NumPy** - 数据处理
- **Matplotlib**, **Seaborn** - 可视化
- **scikit-learn** - 机器学习
- **TensorFlow** / **PyTorch** - 深度学习
- **LSTM**, **RNN**, **BILSTMN** - 时序模型

---
## 🗂️ Project Structure

以下是项目的主要文件和目录结构，便于了解各文件的用途和内容：

```plaintext
├── TSLA.csv                # 📂 原始数据集，包含特斯拉股票的历史数据
├── 数据分析.ipynb           # 📊 数据分析过程，包含数据清洗、可视化和特征工程
├── allmodel.ipynb          # 🧠 模型训练和展示结果，包括模型性能对比和评估
├── RNN_model.py            # 🔄 RNN模型代码，适用于短期趋势预测
├── LSTM_model.py           # ⏳ LSTM模型代码，捕捉长期依赖性
├── stock_mlp_model.py      # 🔍 MLP模型代码，使用全连接层进行特征预测
├── stock_bilstm_model.py    # 🔄 BiLSTM模型代码，双向LSTM捕捉更全面的序列依赖
└── README.md               # 📜 项目简介（当前文件）
```
📂 TSLA.csv
原始的 TSLA 股票数据文件，包括日期、开盘价、收盘价等特征。此数据用于后续的数据分析和模型训练。

📊 数据分析.ipynb
包含对原始数据的清洗、统计分析和可视化，帮助我们更好地理解数据的分布和趋势。

🧠 allmodel.ipynb
模型训练和结果展示的主要文件，包含 RNN、LSTM、MLP 和 BiLSTM 等模型的训练流程以及预测效果对比。此文件通过图表和评估指标帮助直观展示模型表现。

🔄 RNN_model.py & BiLSTM_model.py

RNN_model.py：经典的循环神经网络模型，适用于短期趋势预测。

BiLSTM_model.py：双向 LSTM 模型，能从数据的双向依赖中提取更丰富的特征。

⏳ LSTM_model.py
长短期记忆网络模型，用于捕捉数据中的长期依赖关系，对价格趋势有更好的适应性。

🔍 stock_mlp_model.py
基于多层感知机（MLP）的模型文件，适合从经过特征工程的输入中预测结果。


## 📊 Dataset Overview

本项目使用的数据集为**特斯拉（TSLA）股票历史数据**，包含股票的开盘价、收盘价、最高价、最低价、交易量等特征。该数据集为我们分析和预测股票价格趋势提供了重要的历史信息。

### 数据特征

- **Date**: 日期
- **Open**: 开盘价
- **Close**: 收盘价
- **High**: 最高价
- **Low**: 最低价
- **Volume**: 成交量
- **Money**: 当日交易金额
- **prev_close**: 前一日收盘价
- **daily_return**: 当日收益率，计算方式为 `(Close - prev_close) / prev_close`

数据来自于[数据来源或API链接](https://data-source-link)，数据范围涵盖了特定的时间周期，为构建和评估时序模型提供了足够的信息。

---

## 📈 Data Analysis

在数据分析阶段，我们首先对数据进行了清理和预处理，以确保数据的完整性和一致性。接下来，通过数据的基本统计描述和可视化分析，我们揭示了 TSLA 股票价格的趋势和波动特征。

### 1. 基本统计描述

对关键特征进行了统计分析，以下是一些基本统计数据：

<div align="center">

| Feature       | Mean     | Median   | Min      | Max      |
|---------------|----------|----------|----------|----------|
| Open          | 321.98   | 241.35   | 21.93    | 2295.12  |
| Close         | 322.04   | 241.34   | 21.95    | 2238.75  |
| High          | 329.06   | 245.03   | 23.00    | 2318.49  |
| Low           | 314.64   | 236.59   | 21.50    | 2186.52  |
| Volume        | 2.40e+07 | 7.20e+06 | 2.39e+05 | 3.06e+08 |
| Money         | 7.82e+09 | 1.17e+09 | 0.00e+00 | 1.51e+11 |
| daily_return  | 0.0016   | 0.0012   | -0.7748  | 0.2441   |

</div>


### 2. 数据趋势和季节性

通过对数据的时间序列图分析，我们发现了 TSLA 股票的价格在一定周期内呈现上升趋势，同时存在波动性。以下为价格趋势和波动性图表：

- **价格与指数移动平均线（EMA）图**：展示了 TSLA 股票的开盘价与收盘价，并叠加了短期与长期的指数移动平均线（EMA）。EMA 有助于平滑价格波动，揭示价格的整体趋势方向。

![Stock Price with Exponential Moving Averages](image/Stock_Price_with_and_Exponential_Moving_Averages.png) 


- **布林带与价格波动分析图**：通过布林带展示价格的波动范围，布林带的宽度反映了价格的波动性。高波动期间布林带较宽，低波动期间布林带收窄，有助于识别潜在的价格反转点。

![Bollinger Bands and Price Volatility Analysis](image/Bollinger_Bands_and_Prince_Volatility_Analysis.png) <!-- 替换为布林带图表链接 -->

### 3. 特征相关性

为了进一步理解不同特征之间的关系，我们计算了特征的相关性矩阵。以下是相关性热图：

<div align="center">
    <img src="image/Feature_Correlation_Matrix.png" alt="Correlation Matrix">
</div>

### 4. 夏普率分析

在特征工程中，我们引入了**夏普率**（Sharpe Ratio）这一特征，用于评估收益的风险调整回报。整体夏普率通过平均日收益率与标准差的比值来计算。此外，我们还绘制了一个**30 天滚动夏普率图**，以观察不同时间段内的收益与风险关系。

- **30 天滚动夏普率图**：展示了 30 天滚动窗口的夏普率变化，有助于识别 TSLA 股票在不同市场条件下的风险回报情况。

<div align="center">
    <img src="image/Rolling_30-Day_Sharpe_Ratio_Over_Time.png" alt="Rolling 30-Day Sharpe Ratio Over Time">
</div>

---
---

## 🧮 Model Performance Evaluation

在本项目中，我们使用了多种时序模型来预测 TSLA 股票价格，包括 RNN、LSTM、BiLSTM 和 MLP。为全面评估各模型的性能，我们引入了 **RMSE** (均方根误差)、**MAE** (平均绝对误差) 和 **R²** (决定系数) 作为评估指标，以确保对预测误差和拟合效果的准确把握。

| 📊 模型        | 🔍 RMSE   | 📉 MAE    | 📈 R²      | 📖 链接                      |
|---------------|----------|----------|----------|----------------------------|
| **RNN**       | 75.5083    | 54.5790    | 0.9462    | [查看文档](document/RNN.md)    |
| **LSTM**      | 98.2126    | 74.1667   | 0.9090   | [查看文档](document/LSTM.md)   |
| **BiLSTM**    | 80.8736    | 59.3866    |  0.9383    | [查看文档](document/BILSTM.md) |
| **MLP**       | 80.6509    | 58.6375    | 0.9386    | [查看文档](document/MLP.md)    |

---

这些指标帮助我们量化各模型在预测过程中的误差大小和拟合质量。其中：

- **RMSE (均方根误差)**：衡量模型预测值与实际值之间的标准差，值越小表示预测越准确。
- **MAE (平均绝对误差)**：反映预测值与实际值的平均差异，适合直接评估预测误差的整体水平。
- **R² (决定系数)**：用于评估模型对数据变异的解释程度，值越接近 1 表示模型拟合效果越好。

通过以上评估，我们发现 **RNN** 模型在各项指标上表现最佳，适用于本项目中股票价格的长短期趋势预测。如需查看每个模型的详细训练过程及参数设置，可点击对应的**模型文档链接**进一步了解。

---
