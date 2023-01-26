import pandas as pd

from binance.spot import Spot

# get api 
api = Spot()

# get last 500 trades
trades = api.trades(symbol='BTCUSDT', limit=500)

# convert to dataframe
df = pd.DataFrame(trades)
print(df)

# get last 1 kline 
klines = api.klines(symbol='BTCUSDT', interval='1d', limit=20)
dataframe = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
print(dataframe)

# get moving average of low of the last 7 data points
ma = dataframe['Low'].rolling(7).mean()
print(ma)

