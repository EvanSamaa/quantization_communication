import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Softmax, Input, ThresholdedReLU, Flatten
from tensorflow.keras.activations import sigmoid
import random
def create_MLP_model(input_shape, k):
    # outputs logit
    inputs = Input(shape=input_shape)
    x = perception_model(inputs, k, 5)
    model = Model(inputs, x, name="max_nn")
    return model
def create_MLP_model_with_transform(input_shape, k):
    # needs to call transform first
    # outputs logit
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = perception_model(x, k, 5)
    model = Model(inputs, x, name="max_nn")
    return model
def ranking_transform(x):
    out = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for k in range(x.shape[0]):
        for i in range(0, x.shape[1]):
            for j in range(0, x.shape[1]):
                if x[k, i] >= x[k, j]:
                    out[k, i, j] = 1
    return tf.convert_to_tensor(out, dtype=tf.float32)
def create_uniformed_quantization_model(input_shape, k):
    # outputs logit
    inputs = Input(shape=input_shape)
    x = tf.round(inputs * 1000)/1000
    x = perception_model(x, k, 5)
    model = Model(inputs, x, name="max_nn_with_rounding")
    return model
def create_regression_MLP_netowkr(input_shape, k):
    inputs = Input(shape=input_shape)
    x = perception_model(inputs, 1, 5)
    x = tf.squeeze(x)
    model = Model(inputs, x, name="max_nn_with_regression")
    return model
def create_LSTM_model(k, input_shape=[]):
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.LSTM(10)(inputs)
    x = LeakyReLU()(x)
    x = Dense(20)(x)
    x = LeakyReLU()(x)
    x = Dense(10)(x)
    model = Model(inputs, x, name="max_rnn")
    return model

def create_LSTM_model_backwards(k, input_shape=[]):
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.LSTM(30, go_backwards=True)(inputs)
    x = LeakyReLU()(x)
    x = Dense(20)(x)
    x = LeakyReLU()(x)
    x = Dense(10)(x)
    model = Model(inputs, x, name="max_rnn")
    return model
def create_LSTM_model_with2states(k, input_shape=[], state_size=10):
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.LSTM(30, return_sequences=True)(inputs)
    x = LeakyReLU()(x)
    x = Dense(20)(x)
    x = LeakyReLU()(x)
    x = Dense(10)(x)
    model = Model(inputs, x, name="max_rnn")
    return model
def create_encoding_model(k, l, input_shape):
    inputs = Input(shape=input_shape)
    x_list = tf.split(inputs, num_or_size_splits=k, axis=1)
    encoding = Encoder_module(l)(x_list[0])
    for item in x_list[1:]:
        encoding = tf.concat((encoding, Encoder_module(l)(item)), axis=1)
    out = perception_model(encoding, k, 5)
    model = Model(inputs, out, name="auto_encoder_nn")
    return model

def Encoder_module(L):
    def encoder_module(x):
        x = Dense(20)(x)
        x = LeakyReLU()(x)
        x = Dense(L)(x)
        x = tf.keras.activations.tanh(x) + tf.stop_gradient(tf.math.sign(x) - tf.keras.activations.tanh(x))
        return x
    return encoder_module

# def Encoder_module(x):
#     x = Dense(20)(x)
#     x = LeakyReLU()(x)
#     x = Dense(L)(x)
#     x = tf.keras.activations.tanh(x) + tf.stop_gradient(tf.math.sign(x) - tf.keras.activations.tanh(x))
#     return x


def perception_model(x, output, layer, logit=True):
    for i in range(layer-1):
        x = Dense(50)(x)
        x = LeakyReLU()(x)
    x = Dense(output)(x)
    if logit:
        return x
    else:
        return Softmax(x)



