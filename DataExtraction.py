import os
import pandas as pd

from binance.spot import Spot 
from dotenv import load_dotenv

# get keys
load_dotenv()
api_key = os.getenv('API_KEY')
api_secret = os.getenv('API_SECRET')

# Create a new instance of the API class
api = Spot()

# Get the data
data = api.klines(symbol='BTCUSDT', interval='1d')

# turn data into a dataframe
df = pd.DataFrame(data)
print(df)