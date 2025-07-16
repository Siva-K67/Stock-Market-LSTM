# Stock Market Prediction using LSTM

This project applies Long Short-Term Memory (LSTM) neural networks to forecast stock prices based on historical data. LSTM models are capable of learning order dependence in sequence prediction problems, making them well-suited for time series forecasting in financial markets.

## 📈 Objective

The goal of this notebook is to experiment with deep learning models to predict the next-day closing price of a stock using its historical data (Open, High, Low, Close, Volume).

## 🧠 Model Used

- **LSTM (Long Short-Term Memory)** neural network using TensorFlow/Keras
- Sequential model with multiple LSTM layers and dropout for regularization

## 📊 Dataset

The dataset used is historical stock data obtained from [Yahoo Finance](https://finance.yahoo.com/) or similar financial APIs. It typically includes:
- Date
- Open
- High
- Low
- Close
- Volume

> Note: The dataset can be replaced with any stock ticker of interest.

## ⚙️ Technologies

- Python 3.x
- Pandas, NumPy
- TensorFlow / Keras
- Matplotlib / Seaborn
- scikit-learn (for data scaling)

## 🚀 How to Run

1. Install required libraries:
   ```bash
   pip install -r requirements.txt
