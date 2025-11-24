# ~~ Stock Price Predictor



###  Overview



This project is a machine learning-based stock price prediction system that uses historical stock data to forecast future prices. Leveraging LSTM neural networks, the model identifies patterns in time-series data and predicts stock closing prices for the next day.



### Features



*  Fetches real-time historical stock data using Yahoo Finance.

*  Data preprocessing with normalization and time-windowing.

*  LSTM-based deep learning model for time-series forecasting.

*  Graphical visualization of actual vs predicted stock prices.

*  Next-day price prediction

*  Modular code structure for easy customization.



 ## Technologies / Tools Used



Python 3

NumPy

Pandas

Matplotlib

TensorFlow / Keras

Scikit-learn

yFinance API



### Installation & Running the Project



## 1. Clone or download the repository



```

git clone <your-repo-link>

cd <project-folder>

```


## 2. Install required dependencies



```

pip install numpy pandas matplotlib tensorflow scikit-learn yfinance

```



## 3. Run the project



Use the following command to execute the script:



```

python "Stock Price Predictor (1).py"

```



## Instructions for Testing



* Modify the ticker symbol (default: *GOOGL*) in the script to test prediction for different stocks.

* Adjust the *LOOK_BACK_PERIOD*, *EPOCHS*, or *BATCH_SIZE* to experiment with model accuracy.

* Observe the plotted graph to compare actual and predicted values.

* Check the terminal output for the next-day predicted price.

