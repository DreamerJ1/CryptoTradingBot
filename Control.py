import warnings

from Trading import Trading 
from TradingModel import TradingModel

# turn off warnings
warnings.filterwarnings("ignore")

model = TradingModel('BTC-USD')
# model.data_preprocessing_training()

# train models
rnn_model_size = 7
window = model.create_window_training(rnn_model_size, rnn_model_size, 1)
rnn = model.rnn_model(window, retrain=False, clear_model=True, epochs=2000, learning_rate=0.0001, patience=5)

# window = model.create_window_training(7, 7, 7)
# rnn_multi = model.rnn_multi_step_model(window, retrain=False, clear_model=True, epochs=300, learning_rate=0.001, patience=10)

# create prediction values
# model.data_preprocessing_prediction()

# # predict tomorrow
window = model.create_window_training(rnn_model_size, rnn_model_size, 1)
# prediction = model.predict_tomorrow(rnn, window, plot_type="multi")
prediction, last_price = model.predict_position(rnn, window)

# get value from tensor
print(prediction, last_price)
prediction = prediction.numpy().item()
print(prediction)

# # predict week
# window = model.create_window_training(7, 7, 7)
# # prediction, last_week = model.predict_tomorrow(rnn_multi, window, plot_type="multi")
# prediction, last_week = model.predict_position(rnn_multi, window)

# create trade object
trader = Trading('BTCUSDT', '1d')

# trade a single day
# trader.single_day_trade(trader.convert_to_btc(prediction))
trader.print_orders()