import os
import pandas as pd

from binance.client import Client
from dotenv import load_dotenv

class Trading():
    def __init__(self, symbol, interval):
        # get keys
        load_dotenv()
        api_key = os.getenv('API_KEY')
        api_secret = os.getenv('API_SECRET')

        # create client
        self.client = Client(api_key=api_key, api_secret=api_secret)

        # set variables
        self.symbol = symbol
        self.interval = interval
        self.max_trade_amount = self.convert_to_btc(20)
        self.current_trade_amount = self.convert_to_btc(11)

        print(f"Current trade amount: {self.current_trade_amount:.9f} BTC")

    def single_day_trade(self, prediction):
        # get current price
        current_price = self.get_current_price()

        # check which way to trade
        if current_price > prediction:
            print("Sell")
            self.client.order_limit_sell(
            symbol='BTCUSDT',
            quantity=f"{self.current_trade_amount:.9f}",
            price=prediction
            )           
        else:
            print("Buy")
            self.client.order_limit_buy(
            symbol='BTCUSDT',
            quantity=f"{self.current_trade_amount:.9f}",
            price=prediction
            )    

    def check_orders(self):
        # get orders
        orders = self.client.get_all_orders(symbol=self.symbol)

        # check if order is still open
        if orders[0]['status'] == 'FILLED':
            return False
        else:
            return True  

    def print_orders(self):
        # get orders
        orders = self.client.get_all_orders(symbol=self.symbol)

        # print orders nicely
        for order in orders:
            print(f"{order['side']} {order['origQty']} {order['price']} {order['status']}")
            print()
            
    def check_if_can_buy(self):
        # get account info
        account = self.client.account()

        # check if can buy
        if float(account['balances'][0]['free']) >= self.current_trade_amount:
            return True
        else:
            return False

    def get_current_price(self):
        # get current price
        price = self.client.get_symbol_ticker(symbol=self.symbol)

        # return current price
        return float(price['price'])

    def check_account(self):
        # get account info
        account = self.client.account()

        # print account info
        print(account)

    def convert_to_btc(self, amount):
        # get current price
        current_price = self.get_current_price()

        # convert to btc
        return amount / current_price