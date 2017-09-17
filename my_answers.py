import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
	# The highest index should be 6 indices before the last one
    max_idx = len(series) - window_size

    # containers for input/output pairs
    X = [series[idx:idx + window_size] for idx in range(max_idx)]
    y = series[window_size:]

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    # Instatiating the model
    model = Sequential()

    # Adding the layers to the model
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))

    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    # In one of the exercises I've been able to use more punctiations
    # so I've decided to allow a little bit more punctuaction
    #punctuation = ['!', ',', '.', ':', ';', '?']

    # Defining what is ok to have in the text
    punctuation = "'\"?!,.;:()&$% "
    numbers = "0123456789"
    letters = "abcdefghijklmnopqrstuvwxyz"
    cap_letters = letters.upper()
    allowed_chars = punctuation + numbers + letters + cap_letters

    # Defining unwanted chars
    unwanted_chars = set([ch for ch in text if ch not in allowed_chars])

    for ch in unwanted_chars:
        text = text.replace(ch, ' ')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    pass
