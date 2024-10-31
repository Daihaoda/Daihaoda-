import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer with added dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # Linear layer, first reduce the dimension of LSTM output features
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Activation layer (ReLU)
        self.relu = nn.ReLU()
        
        # Dropout Layer
        self.dropout = nn.Dropout(dropout)
        
        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim // 2)
        
        # The output layer of the last layer
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, x):
        # Initialize hidden states and memory cells
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Output of LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        
        # Select the output of the last time step
        out = out[:, -1, :]
        
        # First layer fully connected + activation + dropout + batch normalization
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.batch_norm(out)
        
        # The last layer is fully connected and outputs the final result
        out = self.fc2(out)
        
        return out


def create_model(input_dim, hidden_dim=512, num_layers=2, output_dim=1):
    model = StockLSTM(input_dim, hidden_dim, num_layers, output_dim)
    return model


from sklearn.metrics import r2_score

def train_model(model, X_train, y_train, X_test, y_test, num_epochs=300, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    r2_train_values = []
    r2_test_values = []

    model.train()
    for epoch in range(num_epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Calculate R² for training data
        r2_train = r2_score(y_train.detach().numpy(), outputs.detach().numpy())
        r2_train_values.append(r2_train)

        # Compute test loss and R² every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                outputs_test = model(X_test)
                loss_test = criterion(outputs_test, y_test)
                r2_test = r2_score(y_test.detach().numpy(), outputs_test.detach().numpy())
                r2_test_values.append(r2_test)
            print(f'Epoch [{epoch + 1}/{num_epochs}], train_Loss: {loss.item():.8f}, test_Loss: {loss_test.item():.8f}, train_R2: {r2_train:.4f}, test_R2: {r2_test:.4f}')
            model.train()  # Set model back to training mode

    return model, losses, r2_train_values, r2_test_values


def plot_losses(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def evaluate_model(model, X_train, X_test, y_train, y_test, scaler):
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train).detach().numpy()
        test_pred = model(X_test).detach().numpy()
    
    train_pred_scaled = scaler.inverse_transform(train_pred)
    test_pred_scaled = scaler.inverse_transform(test_pred)
    y_train_actual = scaler.inverse_transform(y_train.numpy())
    y_test_actual = scaler.inverse_transform(y_test.numpy())
    
    return train_pred_scaled, test_pred_scaled, y_train_actual, y_test_actual


def plot_predictions(y_train_actual, y_test_actual, train_pred_scaled, test_pred_scaled):

    plt.figure(figsize=(10, 6))
    plt.plot(y_train_actual, label='Actual Train')
    plt.plot(train_pred_scaled, label='Predicted Train')
    plt.title('Training Set: Actual vs Predicted')
    plt.legend()
    plt.show()
    

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_actual, label='Actual Test')
    plt.plot(test_pred_scaled, label='Predicted Test')
    plt.title('Test Set: Actual vs Predicted')
    plt.legend()
    plt.show()


    plt.figure(figsize=(10, 6))
    plt.plot(y_test_actual - test_pred_scaled, label='Residuals (Test)')
    plt.title('Residuals of Test Set')
    plt.xlabel('Time Steps')
    plt.ylabel('Residuals')
    plt.legend()
    plt.show()
    
    # 整体对比
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(y_train_actual)), y_train_actual, label='Actual Train')
    plt.plot(np.arange(len(y_train_actual), len(y_train_actual) + len(y_test_actual)), y_test_actual, label='Actual Test')
    plt.plot(np.arange(len(y_train_actual)), train_pred_scaled, label='Predicted Train')
    plt.plot(np.arange(len(y_train_actual), len(y_train_actual) + len(y_test_actual)), test_pred_scaled, label='Predicted Test')
    plt.title('Overall Actual vs Predicted (Train and Test)')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.show()