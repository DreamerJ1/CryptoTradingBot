import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM

from DataExtraction import DataExtraction

class Model:
    def __init__(self) -> None:
        self.train_data = DataExtraction().get_klines()
        self.testData = None
        self.x_train = []
        self.y_train = []
        self.sc = MinMaxScaler()
        
    def data_preprocessing(self):
        # get open price
        self.train_data = self.train_data.iloc[:, 1:2].values

        # scale data
        self.train_data = self.sc.fit_transform(self.train_data)
        
        # split the data into training and testing at 80/20
        training_set_size = int(len(self.train_data) * 0.8)
        self.train_data, self.test_data = self.train_data[0:training_set_size, :], self.train_data[training_set_size:len(self.train_data), :]

        # split the data into x_train and y_train and reshape 
        self.X_train = self.train_data[0:len(self.train_data)-1]
        self.y_train = self.train_data[1:len(self.train_data)]
        self.X_train = np.reshape(self.X_train, (len(self.X_train), 1, 1))

        # split the data into x_test and y_test and reshape
        self.X_test = self.test_data[0:len(self.test_data)-1]
        self.y_test = self.test_data[1:len(self.test_data)]
        self.X_test = np.reshape(self.X_test, (len(self.X_test), 1, 1))

    def train(self):
        # initialize the RNN
        self.regressor = Sequential()
        self.regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))
        self.regressor.add(Dense(units = 1))
        
        # compile the RNN
        self.regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

        # fit the RNN to the training set
        self.regressor.fit(self.X_train, self.y_train, batch_size = 32, epochs = 200)

        # save the model
        self.regressor.save('model.h5')

    def predictTesting(self):
        # try and load model
        try:
            print('Loading model...')
            self.regressor = load_model('model.h5')
        except:
            print('No model found, training now...')
            self.train()

        # predict the test set results and inverse transform
        self.y_pred = self.regressor.predict(self.X_test)
        self.y_pred = self.sc.inverse_transform(self.y_pred)

        # visualize the results
        self.y_test = self.sc.inverse_transform(self.y_test)
        plt.plot(self.y_test, color = 'red', label = 'Real BTC Price')
        plt.plot(self.y_pred[1:], color = 'blue', label = 'Predicted BTC Price')
        plt.title('BTC Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('BTC Price')
        plt.legend()
        plt.show()

        # print the results as a table
        self.y_pred = self.y_pred.reshape(len(self.y_pred), 1)
        self.y_test = self.y_test.reshape(len(self.y_test), 1)
        self.results = np.concatenate((self.y_test, self.y_pred), axis = 1)
        print(self.results)

    def predictTomorrow(self):
        # try and load model
        try:
            print('Loading model...')
            self.regressor = load_model('model.h5')
        except:
            print('No model found, training now...')
            self.train()

        # get real time data of last 10 days
        self.real_time_data = DataExtraction().get_klines(limit=10)

        # get the last 10 days of open price
        self.real_time_data = self.real_time_data.iloc[:, 1:2].values

        # scale the data
        self.real_time_data = self.sc.transform(self.real_time_data)

        # reshape the data
        self.real_time_data = np.reshape(self.real_time_data, (len(self.real_time_data), 1, 1))

        # loop for the next month and predict the each day
        for i in range(0):
            # predict the price of tomorrow
            self.predicted_price = self.regressor.predict(self.real_time_data)
            
            # add the last predicted price to the real time data
            self.real_time_data = np.append(self.real_time_data, self.predicted_price[-1])

            # reshape the real time data
            self.real_time_data = np.reshape(self.real_time_data, (len(self.real_time_data), 1, 1))

            print(len(self.real_time_data))

        # predict the price of tomorrow and inverse transform
        self.predicted_price = self.regressor.predict(self.real_time_data)
        self.predicted_price = self.sc.inverse_transform(self.predicted_price)

        # print the results
        print(self.predicted_price)

        # visualize the results
        self.real_time_data = self.real_time_data.reshape(len(self.real_time_data), 1)
        self.real_time_data = self.sc.inverse_transform(self.real_time_data)
        plt.plot(self.real_time_data, color = 'red', label = 'Real BTC Price')
        plt.plot(self.predicted_price, color = 'blue', label = 'Predicted BTC Price')
        plt.title('BTC Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('BTC Price')
        plt.legend()
        plt.show()
