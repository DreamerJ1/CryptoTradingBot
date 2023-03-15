import numpy as np
import matplotlib.pyplot as plt

from WindowGenerator import WindowGenerator

class WindowGeneratorPrediction(WindowGenerator):
    def __init__(self, input_width, label_width, shift, df_predict, label_columns=None):
        # Store the raw data.
        self.df_predict = df_predict

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(df_predict.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    # Properties
    @property
    def predict(self):
        return self.make_dataset(self.df_predict)

    @property
    def predict_single(self):
        return self.make_dataset(self.df_predict[-7:])

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.predict))
            # And cache it for next time
            self._example = result
        return result
        