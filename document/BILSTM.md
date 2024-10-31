# 🌐 Bidirectional Long Short-Term Memory (BiLSTM) Documentation

---

## 📘 Overview

In this page, we will introduce the **Bidirectional Long Short-Term Memory Network (BiLSTM)** model for predicting TSLA stock prices. Based on the traditional LSTM, BiLSTM captures the contextual information of time series data from both the forward and backward directions to obtain richer feature representations, which is very suitable for time series prediction tasks that require accurate capture of contextual dependencies.
---

## 🧠 Model Architecture

Below is the network architecture of our BiLSTM model, showing the model hierarchy and the number of neurons and parameter settings for each layer:

![BiLSTM Architecture](../image/BiLSTM-model.png) 

- **x_i**: Time step sequence features
- **a_t**: The forward and backward hidden states at time step `t`
- **o_t**: Forecasted value (TSLA stock price)

### 🔍 Key Configuration Parameters
- **Hidden Units**: 512
- **Layers**: 2
- **Bidirectional**: True
- **Optimizer**: Adam
- **Learning Rate**: 0.001

---

## 📊 Performance Metrics

In order to better quantify the prediction effect of the model, we use the following indicators to evaluate the performance of the BiLSTM model:

| metric         | value   |
|--------------|---------|
| RMSE | 80.8736 |
| MAE  | 59.3866 |
| R²    | 0.9383  |

These metrics reflect the prediction accuracy and fit quality of the model in different aspects. **RMSE** and **MAE** measure the prediction bias and error margin of the model respectively, while **R²** shows the model's ability to explain the target variable.
---

## 📈 Training and Validation Curves

In order to observe the stability and generalization ability of model training, we plotted the following curves of training and validation errors versus the number of iterations:

![Training and Validation Curves](../image/BiLSTM-loss.png)

This chart shows the error convergence of the model on the training set and the validation set, which can intuitively determine whether the model is overfitting or underfitting.

---

## 📋 Results Analysis

### 1. **RMSE** 

The RMSE of the BiLSTM model is `80.8736`, indicating that the deviation between the predicted results and the actual values ​​is small, and it is suitable for short-term and long-term prediction of complex time series data.

### 2. **MAE** 

The MAE value is `59.3866`, which further supports the effectiveness of the model in error control.

### 3. **R²** 

The coefficient of determination is `0.9383`, indicating that the model explains 93.83% of the variation of the target variable and has a good fit effect.

---

## 🌟 Experimental visualization results

- **Test set visualization**: The model's predicted trend on the test set is close to the actual trend.
![Test](../image/BiLSTM-test.png)
  
- **Overall prediction results**: The figure below shows the prediction effect of the BiLSTM model on the entire dataset, further illustrating its good time series modeling capabilities.
![Train+Test](../image/BiLSTM-all.png)

---

## 🔗 Access Further Documentation

For more detailed model training, parameter settings, and code examples, please visit the following links:

- [BiLSTM Model](../stock_bilstm_model.py)

---

> **Tips**: The BiLSTM model is suitable for time series analysis tasks that need to capture bidirectional dependencies. The number of hidden layer units can be further adjusted or regularization methods can be added to improve model performance.
