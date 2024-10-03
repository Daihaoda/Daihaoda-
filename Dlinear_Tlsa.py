import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler

# 0. 使用GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(torch.cuda.get_device_name())

# 1. 加载TSLA和SPY的CSV文件
file_path_tsla = r'./TSLA.csv'
file_path_spy = r'./SPY.csv'

data_tsla = pd.read_csv(file_path_tsla)
data_spy = pd.read_csv(file_path_spy)

# 确保日期列转换为 pandas 的 datetime 类型
data_tsla['Date'] = pd.to_datetime(data_tsla['Date'])
data_spy['Date'] = pd.to_datetime(data_spy['Date'])

# 根据 'Date' 列合并数据，这样只有日期匹配的数据才会被保留
merged_data = pd.merge(data_tsla[['Date', 'Close']], data_spy[['Date', 'Close']], on='Date', suffixes=('_TSLA', '_SPY'))



prices_tsla = merged_data['Close_TSLA'].values.astype(np.float32)
prices_spy = merged_data['Close_SPY'].values.astype(np.float32)

# 检查数据是否对齐（如果需要可以做进一步的数据处理）
assert len(prices_tsla) == len(prices_spy), "TSLA和SPY数据长度不一致！"



# 3. 构造训练数据集
TIME_STEP = 5
def create_dataset(tsla_data, spy_data, time_step):
    x, y = [], []
    for i in range(len(tsla_data) - time_step):
        # 合并TSLA和SPY数据，形成多特征输入
        combined_features = np.column_stack((tsla_data[i:(i + time_step)], spy_data[i:(i + time_step)]))
        x.append(combined_features)  # 每个时间步的数据都有两个特征
        y.append(tsla_data[i + time_step])  # 预测目标还是TSLA的价格
    return np.array(x), np.array(y)

x_np, y_np = create_dataset(prices_tsla, prices_spy, TIME_STEP)

# 调整x的shape，使其可以作为RNN的输入
# x的形状将会是 (样本数, 时间步, 特征数=2)，这里特征数是TSLA和SPY
y_np = y_np[:, np.newaxis]  # y仍然是预测目标
print(x_np.shape, y_np.shape)  # (样本数, 时间步=5, 特征数=2)



x_train, x_test, y_train, y_test = train_test_split(x_np, y_np, test_size=0.2, shuffle=False)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)

# 5. 构建DLinear模型
class DLinear(nn.Module):
    def __init__(self, input_size):
        super(DLinear, self).__init__()
        self.linear_trend = nn.Linear(input_size, 1)
        self.linear_season = nn.Linear(input_size, 1)

    def forward(self, x):
        # 将输入分为趋势项和季节性项的预测
        trend = self.linear_trend(x.view(x.size(0), -1))  # 扁平化后送入线性层
        season = self.linear_season(x.view(x.size(0), -1))
        return trend + season

# 超参数
input_size = TIME_STEP * 2  # 2个特征 * 时间步长

# 创建模型
dlinear = DLinear(input_size).to(device)
optimizer = torch.optim.Adam(dlinear.parameters(), lr=0.001)
scheduler = ExponentialLR(optimizer, gamma=0.9999)
loss_func = nn.MSELoss()

# 6. 训练模型
train_losses = []
test_losses = []

total_steps = 100000  # 减少训练轮次为10,000步
for step in range(total_steps):
    dlinear.train()
    prediction = dlinear(x_train.to(device))
    optimizer.zero_grad()
    loss = loss_func(prediction, y_train.to(device))
    loss.backward()
    optimizer.step()
    scheduler.step()

    train_losses.append(loss.item())

    with torch.no_grad():
        dlinear.eval()
        test_prediction = dlinear(x_test.to(device))
        test_loss = loss_func(test_prediction, y_test.to(device)).item()
        test_losses.append(test_loss)

    if step % 1000 == 0:
        print(f'Step: {step}, Train Loss: {loss.item()}, Test Loss: {test_loss}')

plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.title('Loss During Training')
plt.show()
# 用模型对测试集进行预测
with torch.no_grad():
    dlinear.eval()
    test_prediction = dlinear(x_test.to(device)).cpu().numpy()  # 把预测结果转到CPU
    y_test_cpu = y_test.cpu().numpy()  # 把y_test转到CPU

# 将实际股价和预测股价绘制成图
plt.figure(figsize=(12, 6))
plt.plot(y_test_cpu, 'b-', label='Actual Prices')  # 实际股价
plt.plot(test_prediction, 'r-', label='Predicted Prices')  # 预测股价
plt.legend(loc='best')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.show()