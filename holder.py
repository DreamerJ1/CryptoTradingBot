"""
A file to hold code not currently in use.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from Unused.Baseline import Baseline

def createBaseline(self, input_width, label_width, shift):
    # create testing window
    window = self.create_window_training(input_width, label_width, shift)

    # create the baseline and compile the model
    baseline = Baseline(label_index=self.column_indices['position'])
    baseline.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()])

    # print the window generator shape
    print("Input shape:",window.example[0].shape)
    print("Output shape:", baseline(window.example[0]).shape)

    # evaluate the baseline
    self.val_performance['Baseline'] = baseline.evaluate(window.val)
    self.performance['Baseline'] = baseline.evaluate(window.test, verbose=0)

    # plot the baseline
    self.plot_model(baseline, window)

def linear_model(self, input_width, label_width, shift):
    # create testing window
    window = self.create_window_training(input_width, label_width, shift)

    # try and load linear model
    try:
        linear = tf.keras.models.load_model('models/linear_model')
        print("Linear model loaded")
    except:
        print("Linear model not found")
        # create the linear model
        linear = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1)
        ])

        # train and evaluate the model
        history = self.compile_and_fit(linear, window)

        # save the model
        linear.save('models/linear_model')

    # evaluate the model
    self.val_performance['Linear'] = linear.evaluate(window.val)
    self.performance['Linear'] = linear.evaluate(window.test, verbose=0)  

    # plot the model
    self.plot_model(linear, window)

    # visualize weights
    plt.bar(x = range(len(self.train_data.columns)), height=linear.layers[0].kernel[:, 0].numpy())
    axis = plt.gca()
    axis.set_xticks(range(len(self.train_data.columns)))
    _ = axis.set_xticklabels(self.train_data.columns, rotation=90)
    plt.show()

def dense_model(self, input_width, label_width, shift):
    # create testing window
    input_width = label_width 
    window = self.create_window_training(input_width, label_width, shift)

    # try and load dense model
    try:
        dense = tf.keras.models.load_model('models/dense_model')
        print("Dense model loaded")
    except:
        print("Dense model not found")
        # create the dense model
        dense = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Reshape([1, -1])
        ])

        # train and evaluate the model
        history = self.compile_and_fit(dense, window)

        # save the model
        dense.save('models/dense_model')

    # evaluate the model
    self.val_performance['Dense'] = dense.evaluate(window.val)
    self.performance['Dense'] = dense.evaluate(window.test, verbose=0)

    # plot the model
    self.plot_model(dense, window)

def cnn_model(self, conv_width, label_width, shift):
    # create testing window
    input_width = label_width + (conv_width - 1)
    window = self.create_window_training(input_width, label_width, shift)

    # try and load cnn model
    try:
        cnn = tf.keras.models.load_model('models/cnn_model')
        print("CNN model loaded")
    except:
        print("CNN model not found")
        # create the cnn model
        cnn = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=32, kernel_size=(3,), activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1),
        ])

        # train and evaluate the model
        history = self.compile_and_fit(cnn, window)

        # save the model
        cnn.save('models/cnn_model')

    # evaluate the model
    self.val_performance['CNN'] = cnn.evaluate(window.val)
    self.performance['CNN'] = cnn.evaluate(window.test, verbose=0)

    # plot the model
    self.plot_model(cnn, window)