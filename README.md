# Stock Market Prediction using LSTM

This project applies Long Short-Term Memory (LSTM) neural networks to forecast stock closing prices based on historical data. LSTM models excel at learning order dependence in sequence prediction problems, making them well‑suited for time series forecasting in financial markets.

## 📈 Objective

Experiment with deep learning models to predict the next‑day closing price of a stock using its historical data (Open, High, Low, Close, Volume).

## 🧠 Model Architecture

We build a **stacked LSTM** model in TensorFlow/Keras:

- **Input shape:** `(100, 1)` — 100 time steps (window size), single feature (Close price)
- **Layer 1:** `LSTM(50, return_sequences=True)`  
- **Layer 2:** `LSTM(50, return_sequences=True)`  
- **Layer 3:** `LSTM(50)`  
- **Output:** `Dense(1)`  

**Compile settings:**  
- Loss: `mean_squared_error` (MSE)  
- Optimizer: `Adam`

**Training parameters:**  
- Epochs: `100`  
- Batch size: `64`  
- Validation split: training vs. test sets provided via `validation_data=(X_test, Y_test)`

## 📊 Data Preprocessing

1. **Scaling:**  
   - Use `MinMaxScaler(feature_range=(0,1))` to normalize closing prices to (0,1).
2. **Train/Test Split:**  
   - First 80% of data for training, remaining 20% for testing.
3. **Reshaping:**  
   - Reshape data to `[samples, time steps, features]` before feeding into LSTM:
     ```python
     X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
     X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)
     ```

## 📊 Dataset

Historical stock data obtained from [Yahoo Finance](https://finance.yahoo.com/) (or similar API). Typical columns:
- Date
- Open
- High
- Low
- Close
- Volume

> You can replace the dataset with any ticker’s historical data by updating the data loading cell in the notebook.

## ⚙️ Technologies

- Python 3.x  
- pandas, NumPy  
- scikit-learn (for scaling)  
- TensorFlow / Keras  
- Matplotlib / Seaborn  

## 🚀 How to Run

1. **Clone the repo** and `cd` into it:
   ```bash
   git clone https://github.com/Siva-K67/Stock-Market-LSTM.git
   cd Stock-Market-LSTM
2. Install dependencies
   ```bas
   pip install -r requirements.txt

4. Launch the notebook
   ```bas
   jupyter notebook "Stock market LSTM.ipynb"

## 🏆 Results
After training for 100 epochs:
- The model learns to track historical trends and forecasts the next-day closing price.
- Performance (e.g., RMSE) will depend on window size, network architecture, and hyperparameters.

## 🌱 Future Work
- Incorporate news sentiment analysis for richer features.
- Add technical indicators (moving averages, RSI, MACD).
- Tune hyperparameters (learning rate, units, layers).
- Deploy as a web app
