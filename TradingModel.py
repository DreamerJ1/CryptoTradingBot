import os
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

from WindowGeneratorClasses.WindowGeneratorTraining import WindowGeneratorTraining
from WindowGeneratorClasses.WindowGeneratorPrediction import WindowGeneratorPrediction

class TradingModel:
    def __init__(self, ticker):
        self.tickerData = yf.Ticker(ticker)
        # get new data 
        self.data = self.tickerData.history(period='1d', start=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
        # self.data["price"] = (self.data["Open"] + self.data["Close"]) / 2
        self.data["price"] = self.data["Close"]

        # create the data frame
        self.df = self.data[["price"]]

        # create a column for log returns
        self.df["log_returns"] = np.log(self.df["price"] / self.df["price"].shift(1))

        # create a column for position
        self.df["position"] = np.where(self.df["log_returns"] > 0, 1, 0)

        # create lag features
        self.df = self.lag_features(7)

        # add the additional features
        self.df['momentum'] = self.df['log_returns'].rolling(5).mean().shift(1)
        self.df['volatility'] = self.df['log_returns'].rolling(20).std().shift(1)
        self.df['distance'] = (self.df['price'] - self.df['price'].rolling(50).mean()).shift(1)

        # drop the NaN values
        self.df.dropna(inplace=True)

        # create column names
        self.column_indices = {name: i for i, name in enumerate(list(self.df.columns))}
        self.num_features = len(self.column_indices)

        # split data into training val(20% of training) and testing
        self.train_data = self.df[self.df.index < (datetime.now() - timedelta(days=65)).strftime('%Y-%m-%d')].copy()
        self.val_data = self.train_data[self.train_data.index >= (datetime.now() - timedelta(days=115)).strftime('%Y-%m-%d')].copy()
        self.test_data = self.df[self.df.index >= (datetime.now() - timedelta(days=65)).strftime('%Y-%m-%d')].copy()

        # convert to normal distribution
        self.mean, self.std = self.train_data.mean(), self.train_data.std()
        self.train_data_normal = (self.train_data - self.mean) / self.std
        self.val_data_normal = (self.val_data - self.mean) / self.std
        self.test_data_normal = (self.test_data - self.mean) / self.std

        # create the performance dictionaries
        self.val_performance = {}
        self.performance = {}

    def data_processing_trainign(self):
        # get new data 
        self.data = self.tickerData.history(period='1d', start=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
        # self.data["price"] = (self.data["Open"] + self.data["Close"]) / 2
        self.data["price"] = self.data["Close"]

        # create the data frame
        self.df = self.data[["price"]]

        # create a column for log returns
        self.df["log_returns"] = np.log(self.df["price"] / self.df["price"].shift(1))

        # create a column for position
        self.df["position"] = np.where(self.df["log_returns"] > 0, 1, 0)

        # create lag features
        self.df = self.lag_features(7)

        # add the additional features
        self.df['momentum'] = self.df['log_returns'].rolling(5).mean().shift(1)
        self.df['volatility'] = self.df['log_returns'].rolling(20).std().shift(1)
        self.df['distance'] = (self.df['price'] - self.df['price'].rolling(50).mean()).shift(1)

        # drop the NaN values
        self.df.dropna(inplace=True)

        # create column names
        self.column_indices = {name: i for i, name in enumerate(list(self.df.columns))}
        self.num_features = len(self.column_indices)

        # split data into training val(20% of training) and testing
        self.train_data = self.df[self.df.index < (datetime.now() - timedelta(days=65)).strftime('%Y-%m-%d')].copy()
        self.val_data = self.train_data[self.train_data.index >= (datetime.now() - timedelta(days=115)).strftime('%Y-%m-%d')].copy()
        self.test_data = self.df[self.df.index >= (datetime.now() - timedelta(days=0)).strftime('%Y-%m-%d')].copy()

        # convert to normal distribution
        self.mean, self.std = self.train_data.mean(), self.train_data.std()
        self.train_data_normal = (self.train_data - self.mean) / self.std
        self.val_data_normal = (self.val_data - self.mean) / self.std
        self.test_data_normal = (self.test_data - self.mean) / self.std

    def data_preprocessing_prediction(self):
        # get the last 7 days of data
        self.data = self.tickerData.history(period='1d', start=(datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
        self.data["price"] = (self.data["Open"] + self.data["Close"]) / 2
        # self.data["price"] = self.data["Close"]

        # create the data frame
        self.df = self.data[["price"]]

        # create a column for log returns
        self.df["log_returns"] = np.log(self.df["price"] / self.df["price"].shift(1))

        # create a column for position
        self.df["position"] = np.where(self.df["log_returns"] > 0, 1, 0)

        # create lag features
        self.df = self.lag_features(7)

        # add the additional features
        self.df['momentum'] = self.df['log_returns'].rolling(5).mean().shift(1)
        self.df['volatility'] = self.df['log_returns'].rolling(20).std().shift(1)
        self.df['distance'] = (self.df['price'] - self.df['price'].rolling(50).mean()).shift(1)

        # drop the NaN values
        self.df.dropna(inplace=True)

        # create column names
        self.column_indices = {name: i for i, name in enumerate(list(self.df.columns))}
        self.num_features = len(self.column_indices)

        # convert to normal for prediction data
        self.predict_normal = (self.df - self.mean) / self.std

    def rnn_model(self, window, retrain=False, clear_model=False, epochs=100, learning_rate=0.001, patience=5):
        if(clear_model):
            # delete the model
            try:
                os.remove('models/rnn_model')
            except:
                print("RNN multi step model not found")

        # try and load rnn model
        try:
            rnn = tf.keras.models.load_model('models/rnn_model')
            print("RNN model loaded")
        except:
            print("RNN model not found")
            # create the rnn model
            rnn = tf.keras.Sequential([
                tf.keras.layers.LSTM(32, return_sequences=True),
                tf.keras.layers.Dense(units=1)
            ])

            # train and evaluate the model
            history = self.compile_and_fit(rnn, window, epochs=epochs, learning_rate=learning_rate, patience=patience, filepath='models/rnn_model')

            # save the model
            rnn.save('models/rnn_model')

        if(retrain):        
            # train and evaluate the model
            patience = 3
            history = self.compile_and_fit(rnn, window, epochs=epochs, learning_rate=learning_rate, patience=patience/2, filepath='models/rnn_model', retrain=retrain)

            # load best model during retrain
            rnn = tf.keras.models.load_model('models/rnn_model')

        # evaluate the model
        self.val_performance['RNN'] = rnn.evaluate(window.val)
        self.performance['RNN'] = rnn.evaluate(window.test, verbose=0)

        # plot the model
        self.plot_model(rnn, window)

        # return the model
        return rnn

    def rnn_multi_step_model(self, window, retrain=False, clear_model=False, epochs=100, learning_rate=0.001, patience=5):
        if(clear_model):
            # delete the model
            try:
                os.remove('models/rnn_multi_step_model')
            except:
                print("RNN multi step model not found")

        # try and load rnn model
        try:
            rnn = tf.keras.models.load_model('models/rnn_multi_step_model')
            print("RNN model loaded")
        except:
            print("RNN model not found")
            # create the rnn model
            rnn = tf.keras.Sequential([
                tf.keras.layers.LSTM(32, return_sequences=False),
                tf.keras.layers.Dense(window.shift*self.num_features, kernel_initializer=tf.initializers.zeros()),
                tf.keras.layers.Reshape([window.shift, self.num_features])
            ])

            # train and evaluate the model
            history = self.compile_and_fit(rnn, window, epochs=epochs, learning_rate=learning_rate, patience=patience, filepath='models/rnn_multi_step_model')

            # save the model
            rnn.save('models/rnn_multi_step_model')

        if(retrain):
            # train and evaluate the model
            history = self.compile_and_fit(rnn, window, epochs=epochs, learning_rate=learning_rate, patience=patience/2, filepath='models/rnn_multi_step_model', retrain=retrain)

            # load best model during retrain
            rnn = tf.keras.models.load_model('models/rnn_multi_step_model')

        # evaluate the model
        self.val_performance['RNN_MULTI'] = rnn.evaluate(window.val)
        self.performance['RNN_MULTI'] = rnn.evaluate(window.test, verbose=0)

        # plot the model
        self.plot_model(rnn, window)

        # return the model
        return rnn

    def predict_position(self, model, window):
        # get last weeks data from prediction
        last_week = self.test_data_normal[-window.label_width:]
        print(last_week)

        # convert normal to tensor
        last_week = tf.convert_to_tensor(last_week)

        # reshape last week 
        last_week = tf.reshape(last_week, [1, window.label_width, self.num_features])

        # get the prediction
        prediction = model.predict(last_week)

        # plot the prediction
        self.plot_positions(last_week, prediction, window)

        # calculate the increase
        self.calculate_increase(prediction[0, :, 0], last_week[0, :, 2])

        # turn prediction into tensor and return it
        return tf.convert_to_tensor(prediction[0, :, 0])[-1], (last_week[0, :, 0][-1] * self.std[0] + self.mean[0])

    def calculate_increase(self, prediction, last_week):
        increase = 0

        # loop over the predictions 
        for i in range(len(prediction)-1):
            # check if last price increase or decresed 
            if(last_week[i+1] < 0 and prediction[i] < 0):
                increase += 1
            elif(last_week[i+1] > 0 and prediction[i] > 0):
                increase += 1
        print("Increase", increase, "out of", len(prediction)-1)

    def plot_positions(self, input, predictions, window):
        # print working values
        print("Input", input[0, :, 2])
        print("Predictions", predictions[0, :, 0])

        # create the figure
        plt.figure(figsize=(12, 8))
        plt.ylabel("Predictions")

        # plot the predictions
        plt.plot(window.input_indices, input[0, :, 2], label='Input', marker='.', zorder=-10)
        plt.scatter(window.label_indices, predictions[0, :, 0], edgecolors='k', label='Predictions', c='#2ca02c', s=64, marker='x')

        # legend and show 
        plt.legend()
        plt.show()    

    def predict_tomorrow(self, model, window, plot_type="normal"):
        # get last weeks data from prediction
        last_week = self.test_data_normal[-window.label_width:]

        # convert normal to tensor
        last_week = tf.convert_to_tensor(last_week)

        # reshape last week 
        last_week = tf.reshape(last_week, [1, window.label_width, self.num_features])

        # get the prediction
        prediction = model.predict(last_week)

        # plot the prediction
        if(plot_type == "normal"):
            self.plot_predictions(last_week, prediction, window)

            # denormalize the predictions
            holder = prediction[0, :, 0]
            values = []
            for i in range(len(holder)):
                values.append(holder[i] * self.std[0] + self.mean[0])
            prediction = tf.concat(values, axis=0)

        elif(plot_type == "multi"):
            # denormalize the predictions
            holder = last_week[0, :, 0]
            values = []
            for i in range(len(holder)):
                values.append(holder[i] * self.std[0] + self.mean[0])
            last_week = tf.concat(values, axis=0)

            holder = prediction[0, :, 0]
            values = []
            for i in range(len(holder)):
                values.append(holder[i] * self.std[0] + self.mean[0])
            prediction = tf.concat(values, axis=0)

            self.plot_predictions(last_week, prediction, window)

        # return the prediction and the last week
        return prediction[-1]

    def plot_predictions(self, input, predictions, window):
        # print working values
        print("Input", input)
        print("Predictions", predictions)

        # create the figure
        plt.figure(figsize=(12, 8))
        plt.ylabel("Predictions")

        # plot the predictions
        plt.plot(window.input_indices, input, label='Input', marker='.', zorder=-10)
        plt.scatter(window.label_indices, predictions, edgecolors='k', label='Predictions', c='#2ca02c', s=64, marker='x')

        # legend and show 
        plt.legend()
        plt.show()    

    def compile_and_fit(self, model, window, patience=5, epochs=100, learning_rate=0.001, filepath='models/model_checkpoint', retrain=False):
        # early stopping and model checkpoint
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, save_best_only=True, monitor='val_loss', mode='min')

        model.compile(loss=tf.losses.MeanAbsoluteError(), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=[tf.metrics.MeanAbsoluteError()])

        if(retrain):            
            history = model.fit(window.train, epochs=epochs, validation_data=window.val, callbacks=[early_stopping, model_checkpoint])    
        else:
            history = model.fit(window.train, epochs=epochs, validation_data=window.val, callbacks=[early_stopping])
            # history = model.fit(window.train, epochs=epochs, validation_data=window.val)

        return history

    def plot_model(self, model, window):
        # plot the model
        window.plot(model)
        plt.show()

    def create_window_training(self, input_width, label_width, shift):
        # create the window
        window = WindowGeneratorTraining(input_width=input_width, label_width=label_width, shift=shift, train_df=self.train_data_normal, val_df=self.val_data_normal, test_df=self.test_data_normal, label_columns=['position'])
        
        # print the window
        print('Window')
        print(window)
        print('Input shape:', window.example[0].shape)
        print('Output shape:', window.example[1].shape)

        return window

    def create_window_prediction(self, input_width, label_width, shift):
        # create the window
        window = WindowGeneratorPrediction(input_width=input_width, label_width=label_width, shift=shift, df_predict=self.predict_normal, label_columns=['position'])
        
        # print the window
        print('Window')
        print(window)
        print('Input shape:', window.example[0].shape)
        print('Output shape:', window.example[1].shape)

        return window

    def lag_features(self, lag_num):
        for lag_num in range(1, lag_num + 1):
            self.df[f'lag_{lag_num}'] = self.df['log_returns'].shift(lag_num)
            self.df.dropna(inplace=True)
        return self.df
