import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 0.使用GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(torch.cuda.get_device_name())
else:
    device = torch.device('cpu')

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

# 1. 加载TSLA和SPY的CSV文件
file_path_tsla = r'./TSLA.csv'
file_path_spy = r'./SPY.csv'
file_path_VIX = r'./VIX_History.csv'

data_tsla = pd.read_csv(file_path_tsla)
data_spy = pd.read_csv(file_path_spy)
data_VIX = pd.read_csv(file_path_VIX)
# 确保日期列转换为 pandas 的 datetime 类型
data_tsla['Date'] = pd.to_datetime(data_tsla['Date'])
data_spy['Date'] = pd.to_datetime(data_spy['Date'])
data_VIX['Date'] = pd.to_datetime(data_VIX['Date'])

# tsla在2020年8月31日以5:1拆股，2020年8月25日以3:1拆股
first_split_mask = data_tsla['Date'] >= pd.to_datetime('2020-08-31')
data_tsla['Close'].loc[first_split_mask] = data_tsla['Close'].loc[first_split_mask] * 5
second_split_mask = data_tsla['Date'] >= pd.to_datetime('2022-08-25')
data_tsla['Close'].loc[second_split_mask] = data_tsla['Close'].loc[second_split_mask] * 3

# 根据 'Date' 列合并数据，这样只有日期匹配的数据才会被保留
merged_data = pd.merge(data_tsla[['Date', 'Close']], data_spy[['Date', 'Close']], on='Date', suffixes=('_TSLA', '_SPY'))
merged_data = pd.merge(merged_data, data_VIX[['Date', 'Close']], on='Date')
merged_data.rename(columns={'Close': 'Close_VIX'}, inplace=True)
# 加载TSLA的Volume数据，并合并到现有数据中
merged_data = pd.merge(merged_data, data_tsla[['Date', 'Volume']], on='Date')
merged_data.rename(columns={'Volume': 'Volume_TSLA'}, inplace=True)

# 将 TSLA, SPY, VIX 和 Volume 作为输入特征
prices_tsla = merged_data['Close_TSLA'].values.astype(np.float32)
prices_spy = merged_data['Close_SPY'].values.astype(np.float32)
prices_vix = merged_data['Close_VIX'].values.astype(np.float32)
volume_tsla = merged_data['Volume_TSLA'].values.astype(np.float32)

# 检查数据是否对齐
assert len(prices_tsla) == len(prices_spy) == len(prices_vix) == len(volume_tsla), "TSLA, SPY, VIX 和 Volume 数据长度不一致！"

# 3. 构造训练数据集
TIME_STEP = 5
# 计算增幅百分比
tsla_change = np.concatenate((np.array([0.0]), (prices_tsla[1:] - prices_tsla[:-1]) / prices_tsla[:-1]))
spy_change = np.concatenate((np.array([0.0]), (prices_spy[1:] - prices_spy[:-1]) / prices_spy[:-1]))
vix_change = np.concatenate((np.array([0.0]), (prices_vix[1:] - prices_vix[:-1]) / prices_vix[:-1]))
volume_change = np.concatenate((np.array([0.0]), (volume_tsla[1:] - volume_tsla[:-1]) / volume_tsla[:-1]))

# 构造数据集的函数
def create_dataset(tsla_change, spy_change, vix_change, volume_change, time_step):
    x, y = [], []
    for i in range(len(tsla_change) - time_step):
        # 每个样本由前5天的变化百分比组成
        combined_features = np.column_stack((tsla_change[i:(i + time_step)],
                                             spy_change[i:(i + time_step)],
                                             vix_change[i:(i + time_step)],
                                             volume_change[i:(i + time_step)]))
        x.append(combined_features)  # 形状：(时间步=5, 特征数=4)
        # 将第6天相对于第5天的TSLA增幅作为预测目标
        y.append(tsla_change[i + time_step])
    return np.array(x), np.array(y)

# 调用函数，生成数据集
x_np, y_np = create_dataset(tsla_change, spy_change, vix_change, volume_change, TIME_STEP)

# 将 y 转换为列向量，以符合模型的输入格式
y_np = y_np[:, np.newaxis]
print(x_np.shape, y_np.shape)  # 应该是 (样本数, 时间步=5, 特征数=4) 和 (样本数, 1)


print(np.sum(np.power(x_np[:,-1] - y_np, 2))/x_np.shape[0])
# 4. 数据集划分为训练集和测试集
from sklearn.model_selection import train_test_split
# 首先将数据划分为训练集（80%）和剩余数据集（20%）
x_train, x_rem, y_train, y_rem = train_test_split(x_np, y_np, test_size=0.2, shuffle=False, random_state=seed)

# 将剩余的20%再划分为验证集和测试集（各10%）
x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem, test_size=0.5, shuffle=False, random_state=seed)

x_train = torch.as_tensor(x_train, dtype=torch.float32)
y_train = torch.as_tensor(y_train, dtype=torch.float32)
x_val = torch.as_tensor(x_val, dtype=torch.float32)
y_val = torch.as_tensor(y_val, dtype=torch.float32)
x_test = torch.as_tensor(x_test, dtype=torch.float32)
y_test = torch.as_tensor(y_test, dtype=torch.float32)

## 超参数 hyperparameter optimisation
# layers = [1, 2, 3]
# hidden_sizes = [10, 50, 100, 200]
layers = [3]
hidden_sizes = [200]
# 记录最好的模型和损失
best_model = None
best_train_loss = float('inf')
best_test_loss = float('inf')

# 遍历不同的layer和hidden size组合
for layer in layers:
    for hidden_size in hidden_sizes:
        print(f'Training RNN with {layer} layers and hidden size {hidden_size}')

        # 5. 构建RNN模型
        class StockPriceRNN(nn.Module):
            def __init__(self, hidden_size, num_layers):
                super(StockPriceRNN, self).__init__()
                self.rnn = nn.RNN(
                    input_size=4,
                    hidden_size=hidden_size,  # 隐藏层单元数
                    num_layers=num_layers,    # RNN层数
                    batch_first=True,
                )
                self.relu = nn.ReLU()
                self.out = nn.Linear(hidden_size, 1)

            def forward(self, x, h_state):
                r_out, h_state = self.rnn(x, h_state)
                outs = self.relu(r_out)
                outs = self.out(outs[:, -1, :])
                return outs, h_state

        # 创建模型
        rnn = StockPriceRNN(hidden_size, layer).to(device)
        # rnn.load_state_dict(torch.load('./min_test_loss_model.pt'))
        optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
        scheduler = ExponentialLR(optimizer, gamma=0.9999)
        loss_func = nn.MSELoss()

        # 创建两个空列表，用于记录训练和测试的损失
        train_losses = []
        test_losses = []

        # 初始化最小训练损失和测试损失
        min_train_loss = float('inf')
        min_test_loss = float('inf')

        # 6. 训练模型
        total_steps = 100000  # 总步数
        record_start = total_steps - 1000  # 记录最后1,000步的起始点

        for step in range(total_steps):  # 增加训练轮次
            rnn.train()

            h_state = None  # 初始化隐藏状态
            prediction, h_state = rnn(x_train.to(device), h_state)

            optimizer.zero_grad()
            loss = loss_func(prediction, y_train.to(device))

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss = loss.item()

            # 验证集评估
            with torch.no_grad():
                rnn.eval()
                val_prediction, h_state = rnn(x_val.to(device), None)  # 在验证集上进行预测
                val_loss = loss_func(val_prediction, y_val.to(device)).item()  # 计算验证集上的损失

                test_prediction, h_state = rnn(x_test.to(device), None)  # 在测试集上进行预测
                test_loss = loss_func(test_prediction, y_test.to(device)).item()  # 计算测试集上的损失
                # if step == 5000:
                #     assert False
            if step >= record_start:
                train_losses.append(train_loss)
                test_losses.append(test_loss)

            if train_loss < min_train_loss:
                min_train_loss = train_loss

            if test_loss < min_test_loss:
                min_test_loss = test_loss
                # if step + 1 % 100 == 0:
                #     torch.save(rnn.state_dict(), "./min_test_loss_model.pt") #每次从最佳开始跑
                #     print('model saved')

            # 每1000步输出训练、验证和测试集上的损失
            if step % 1000 == 0:
                print(f'Step: {step}, Train_Loss: {train_loss}, Val_Loss: {val_loss}, Test_Loss: {test_loss}')

        print(f'Final Minimum Train Loss: {min_train_loss}')
        print(f'Final Minimum Test Loss: {min_test_loss}')

        # 更新最佳模型
        if min_test_loss < best_test_loss:
            best_model = rnn
            best_train_loss = min_train_loss
            best_test_loss = min_test_loss

# 打印最好的模型结果
print(f'Best model with {best_train_loss} train loss and {best_test_loss} test loss')

# # 绘制训练过程中的预测结果
# plt.figure(figsize=(12, 6))
# plt.plot(range(len(y_test)), y_test.cpu().numpy(), 'b-', label='True Prices')
# plt.plot(range(len(test_prediction)), test_prediction.cpu().numpy(), 'r-', label='Predicted Prices')
# plt.legend(loc='best')
# plt.title(f'Step: {step} - Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.show()
#
# plt.figure(figsize=(12, 6))
# plt.plot(train_losses, label='Train Loss')
# plt.plot(test_losses, label='Test Loss')
# plt.legend(loc='best')
# plt.title('Loss During Training')
# plt.xlabel('Steps')
# plt.ylabel('Loss')
# plt.show()
#

# 画出图
# 两个空的列表，一个训练，一个测试记录loss





