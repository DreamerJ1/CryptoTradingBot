import os
import pandas as pd

from binance.spot import Spot
from dotenv import load_dotenv

# get keys
load_dotenv()
api_key = os.getenv('API_KEY')
api_secret = os.getenv('API_SECRET')

# create client
client = Spot(api_key=api_key, api_secret=api_secret)

print(client.account())