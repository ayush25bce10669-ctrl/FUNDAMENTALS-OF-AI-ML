import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

TICKER_SYMBOL = 'GOOGL'
START_DATE = '2018-01-01'
END_DATE = '2023-01-01'
LOOK_BACK_PERIOD = 60
EPOCHS = 25
BATCH_SIZE = 32


def fetch_and_preprocess_data(ticker, start, end, look_back):
    try:
        raw_data = yf.download(ticker, start=start, end=end)['Close']
        dataset = raw_data.values.reshape(-1, 1)
    except Exception as e:
        print("ERROR: Could not fetch data for {}. Check ticker or internet connection.".format(ticker))
        raise e

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    X, Y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        Y.append(scaled_data[i, 0])

    X, Y = np.array(X), np.array(Y)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, Y, scaler, raw_data


def build_and_train_model(X_train, Y_train, look_back, epochs, batch_size):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    return model


def evaluate_and_plot(model, X_test, scaler, raw_data, look_back):
    predicted_prices_scaled = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices_scaled)

    train_data_size = len(raw_data) - len(X_test) - look_back
    test_data_start_index = look_back + train_data_size
    testing_data = raw_data[test_data_start_index:]

    plt.figure(figsize=(14, 7))
    plt.plot(testing_data.index, testing_data.values, label='Actual Price', color='blue')
    plt.plot(testing_data.index, predicted_prices, label='Predicted Price', color='red')
    plt.title(TICKER_SYMBOL)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    last_look_back_days = raw_data.values[-look_back:].reshape(-1, 1)
    last_look_back_days_scaled = scaler.transform(last_look_back_days)
    X_next = np.array([last_look_back_days_scaled])
    X_next = np.reshape(X_next, (X_next.shape[0], X_next.shape[1], 1))

    predicted_next_day_scaled = model.predict(X_next)
    predicted_next_day = scaler.inverse_transform(predicted_next_day_scaled)

    print(predicted_next_day[0, 0])


if __name__ == '__main__':
    X, Y, scaler, raw_data = fetch_and_preprocess_data(
        TICKER_SYMBOL, START_DATE, END_DATE, LOOK_BACK_PERIOD
    )

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

    trained_model = build_and_train_model(
        X_train, Y_train, LOOK_BACK_PERIOD, EPOCHS, BATCH_SIZE
    )

    evaluate_and_plot(trained_model, X_test, scaler, raw_data, LOOK_BACK_PERIOD)