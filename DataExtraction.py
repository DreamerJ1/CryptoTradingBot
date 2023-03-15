import pandas as pd

from binance.spot import Spot

class DataExtraction:
    def __init__(self):
        self.api = Spot()
        
    def get_trades(self, symbol='BTCUSDT', limit=1000):
        trades = self.api.trades(symbol=symbol, limit=limit)
        df = pd.DataFrame(trades)
        print(df)
        
    def get_klines(self, symbol='BTCUSDT', interval='1d', limit=500):
        klines = self.api.klines(symbol=symbol, interval=interval, limit=limit)
        dataframe = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        return dataframe


