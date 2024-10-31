import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class StockBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(StockBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Define bidirectional LSTM
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # The dimension of the bidirectional LSTM output is hidden_dim * 2, because there are two directions
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.bilstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


from sklearn.metrics import r2_score


def train_model(model, X_train, y_train, X_test, y_test, num_epochs=300, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    r2_test_values = []

    model.train()
    for epoch in range(num_epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Compute and print test loss and R² every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():  # Disable gradient calculation for test data
                outputs_test = model(X_test)
                loss_test = criterion(outputs_test, y_test)

                # Calculate R² for test data
                r2_test = r2_score(y_test.detach().numpy(), outputs_test.detach().numpy())
                r2_test_values.append(r2_test)

            print(
                f'Epoch [{epoch + 1}/{num_epochs}], train_Loss: {loss.item():.8f}, test_Loss: {loss_test.item():.8f}, test_R2: {r2_test:.4f}')
            model.train()  # Set model back to training mode

    return losses, r2_test_values


def plot_losses(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def evaluate_model(model, X_train, X_test, y_train, y_test, close_scaler):
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train).detach().numpy()
        test_pred = model(X_test).detach().numpy()

    train_pred_scaled = close_scaler.inverse_transform(train_pred)
    test_pred_scaled = close_scaler.inverse_transform(test_pred)
    y_train_actual = close_scaler.inverse_transform(y_train.numpy())
    y_test_actual = close_scaler.inverse_transform(y_test.numpy())
    
    return y_train_actual, train_pred_scaled, y_test_actual, test_pred_scaled

def plot_predictions(y_train_actual, train_pred_scaled, y_test_actual, test_pred_scaled):
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

def plot_residuals(y_test_actual, test_pred_scaled):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_actual - test_pred_scaled, label='Residuals (Test)')
    plt.title('Residuals of Test Set')
    plt.xlabel('Time Steps')
    plt.ylabel('Residuals')
    plt.legend()
    plt.show()

def plot_overall_comparison(y_train_actual, train_pred_scaled, y_test_actual, test_pred_scaled):
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
