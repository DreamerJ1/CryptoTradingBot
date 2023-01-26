import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Sequential
from binance.spot import Spot

# Instantiate the Spot client
spot = Spot()

# Define the symbol and the interval
symbol = "BTCUSDT"
interval = "1d"

# Retrieve the klines
klines = spot.klines(symbol=symbol, interval=interval, limit=50)
klines = [[float(y) for y in x] for x in klines]
headers = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
df = pd.DataFrame(klines, columns=headers)

# Define the columns to use as features
features = ['Open', 'Close', 'Volume', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume']

# Scale the features to be between 0 and 1
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Split the data into train and test sets
train_data = df[:int(df.shape[0]*0.8)]
test_data = df[int(df.shape[0]*0.8):]

# Define the input and output data for the model
X_train = train_data[features]
y_high_train = train_data[['High']]
y_low_train = train_data[['Low']]
X_test = test_data[features]
y_high_test = test_data[['High']]
y_low_test = test_data[['Low']]

# reshape the input data
X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the LSTM model for high values
high_model = Sequential()
high_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
high_model.add(LSTM(units=50))
high_model.add(Dense(1))
high_model.compile(optimizer='adam', loss='mean_squared_error')
high_model.fit(X_train, y_high_train, epochs=100, batch_size=32)

# Define the LSTM model for low values
low_model = Sequential()
low_model.add(LSTM(units=50, input_shape=(X_train.shape[1], 1)))
low_model.add(Dense(1))
low_model.compile(optimizer='adam', loss='mean_squared_error')
low_model.fit(X_train, y_low_train, epochs=100, batch_size=32)

# Make predictions on the test data
y_high_pred = high_model.predict(X_test)
y_low_pred = low_model.predict(X_test)

# Rescale the predictions
y_high_pred = scaler.inverse_transform(y_high_pred)
y_low_pred = scaler.inverse_transform(y_low_pred)

# Print the predictions
print(y_high_pred)
print(y_low_pred)