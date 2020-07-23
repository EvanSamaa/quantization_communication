import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Softmax, Input, ThresholdedReLU, Flatten
from tensorflow.keras.activations import sigmoid
import random
from util import *

############################## Trained Loss Functions ##############################
def MLP_loss_function(inputshape=[1000, 3]):
    inputs = Input(shape=inputshape)
    x = tf.keras.layers.Reshape((3000, ))(inputs)
    x = Dense(500)(x)
    x = LeakyReLU()(x)
    x = Dense(500)(x)
    x = LeakyReLU()(x)
    x = Dense(1)(x)
    model = Model(inputs, x, name="category_count_MLP")
    return model
def LSTM_loss_function(k, input_shape=[], state_size=30):
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.LSTM(state_size)(inputs)
    x = Dense(60)(x)
    x = LeakyReLU()(x)
    x = Dense(20)(x)
    x = LeakyReLU()(x)
    x = Dense(1)(x)
    model = Model(inputs, x, name="category_count_LSTM")
    return model
def Convnet_loss_function(input_shape = [1000, 4], combinations = 8):
    # you need to format the input into (1, batchsize, encoding size)
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(filters=combinations*2, kernel_size=input_shape[1], strides=1, name="variation_scanners_1")(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=10)(x)
    x = tf.keras.layers.Conv1D(filters=combinations*2, kernel_size=input_shape[1], strides=1, name="variation_scanners_2")(x)
    x = tf.keras.layers.MaxPool1D(pool_size=5)(x)
    x = tf.keras.layers.Reshape([combinations * 19 *2])(x)
    x = tf.keras.layers.Dense(200)(x)
    x = sigmoid(x)
    x = tf.keras.layers.Dense(200)(x)
    x = sigmoid(x)
    x = tf.keras.layers.Dense(1)(x)
    x = LeakyReLU()(x)
    model = Model(inputs, x, name="category_count_conv_net")
    print(model.summary())
    return model
############################## analytical model ##############################
def create_uniformed_quantization_model(k, bin_num=10, prob=True):
    def uniformed_quantization_prob(x):
        x = tf.round(x*bin_num)/bin_num
        max_x = tf.argmax(x, axis=1).numpy()
        max_x = max_x.flatten()
        col = np.arange(0, x.shape[0])
        out = np.zeros(x.shape)
        out[col, max_x] = 1
        out = tf.convert_to_tensor(out)
        return out
    def uniformed_quantization_reg(x):
        x = tf.round(x*bin_num)/bin_num
        max_x = tf.argmax(x, axis=1).numpy()
        return max_x
    if prob:
        return uniformed_quantization_prob
    else:
        return uniformed_quantization_reg
def create_optimal_model_k_2(k, input):
    x = floatbits_to_float(input)



############################## MLP models ##############################
def create_MLP_model(input_shape, k):
    # outputs logit
    inputs = Input(shape=input_shape)
    x = perception_model(inputs, k, 5)
    model = Model(inputs, x, name="max_nn")
    return model
def create_MLP_mean0_model(input_shape, k):
    # outputs logit
    inputs = Input(shape=input_shape)
    x = inputs - 0.5
    x = perception_model(x, k, 5)
    model = Model(inputs, x, name="max_nn")
    return model
def create_large_MLP_model(input_shape, k):
    inputs = Input(shape=input_shape)
    x = perception_model(inputs, 50, 10)
    x = LeakyReLU()(x)
    x = Dense(10)(x)
    model = Model(inputs, x, name="Deep_max_nn")
    print(model.summary())
    return model
def create_MLP_model_with_transform(input_shape, k):
    # model = create_MLP_model_with_transform((k,k), k)
    # needs to call transform first
    # outputs logit
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = perception_model(x, k, 2)
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
def create_regression_MLP_netowkr(input_shape, k):
    inputs = Input(shape=input_shape)
    x = perception_model(inputs, 1, 5)
    x = tf.squeeze(x)
    model = Model(inputs, x, name="max_nn_with_regression0")
    return model
############################## LSTM models ##############################
# model = create_LSTM_model(k, [k, 1], 10)
def create_LSTM_model(k, input_shape=[], state_size=10):
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.LSTM(state_size)(inputs)
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
    x, final_memory_state, final_carry_state = tf.keras.layers.LSTM(30, return_sequences=True, return_state=True)(inputs)
    x = tf.keras.layers.Flatten()(x[:, -2:]) # looking only at the last 2 layers
    x = LeakyReLU()(x)
    x = Dense(20)(x)
    x = LeakyReLU()(x)
    x = Dense(10)(x)
    model = Model(inputs, x, name="max_rnn")
    return model
def create_BLSTM_model_with2states(k, input_shape=[], state_size=10):
    inputs = Input(shape=input_shape)
    x, final_memory_state, final_carry_state = tf.keras.layers.LSTM(30, go_backwards=True, return_sequences=True, return_state=True)(inputs)
    x_1 = tf.keras.layers.Flatten()(x[:, -1:]) # looking only at the last 2 layers
    x = tf.concat((tf.keras.layers.Flatten()(x[:, -1]), tf.keras.layers.Flatten()(x[:, 0])), axis=1)
    x = LeakyReLU()(x)
    x = Dense(20)(x)
    x = LeakyReLU()(x)
    x = Dense(10)(x)
    model = Model(inputs, x, name="max_rnn")
    return model
############################## Encoding models ##############################
def create_encoding_model(k, l, input_shape):
    inputs = Input(shape=input_shape)
    x_list = tf.split(inputs, num_or_size_splits=k, axis=1)
    encoding = Encoder_module(l)(x_list[0])
    for item in x_list[1:]:
        encoding = tf.concat((encoding, Encoder_module(l)(item)), axis=1)
    out = perception_model(encoding, k, 5)
    model = Model(inputs, out, name="auto_encoder_nn")
    return model
def create_uniform_encoding_model(k, l, input_shape):
    inputs = Input(shape=input_shape)
    x_list = tf.split(inputs, num_or_size_splits=k, axis=1)
    encoder_module = Uniform_Encoder_module(1, l, (1,))
    encoding = encoder_module(x_list[0])
    for item in x_list[1:]:
        encoding = tf.concat((encoding, encoder_module(item)), axis=1)
    out = perception_model(encoding, k, 5)
    model = Model(inputs, out, name="auto_encoder_nn")
    return model
def boosting_regression_model(models, input_shape, k):
    inputs = Input(shape=input_shape)
    x = models[0](inputs)
    print(x.shape)
    for model in models[1:]:
        x = tf.concat((x, model(inputs)), axis=1)
    initializer = tf.keras.initializers.Constant(1./2)
    x = Dense(1, kernel_initializer=initializer)(x)
    model = Model(inputs, x, name="ensemble")
    return model
def Encoder_module(L):
    def encoder_module(x):
        x = Dense(20)(x)
        x = LeakyReLU()(x)
        x = Dense(L)(x)
        # x = tf.keras.activations.tanh(x) + tf.stop_gradient(tf.math.sign(x) - tf.keras.activations.tanh(x))
        # x = sign_relu_STE(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = hard_tanh(x) + tf.stop_gradient(tf.sign(x) - hard_tanh(x))
        return x
    return encoder_module
def Uniform_Encoder_module(k, l, input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(20)(inputs)
    x = LeakyReLU()(x)
    x = Dense(l)(x)
    x = tf.keras.activations.sigmoid(x) + tf.stop_gradient(tf.math.sign(x) - tf.keras.activations.sigmoid(x))
    model = Model(inputs, x, name="encoder_unit")
    return model
def create_encoding_model_with_annealing(k, l, input_shape):
    inputs = Input(shape=input_shape)
    epoch = inputs[:, 1][0]
    inputs_mod = inputs[:, 1:]
    print(inputs.shape)
    x_list = tf.split(inputs_mod, num_or_size_splits=k, axis=1)
    encoding = Encoder_module_annealing(l)(x_list[0], epoch)
    for item in x_list[1:]:
        encoding = tf.concat((encoding, Encoder_module_annealing(l)(item, epoch)), axis=1)
    out = perception_model(encoding, k, 5)
    model = Model(inputs, out, name="k2_L2_annealing_nn")
    print(model.summary())
    return model
def Uniform_Encoder_module_with_annealing(k, l, input_shape):
    inputs = Input(shape=input_shape)
    epoch = inputs[:, 1][0]
    inputs_mod = inputs[:, 1:]
    x = Dense(20)(inputs_mod)
    x = LeakyReLU()(x)
    x = Dense(l)(x)
    x = annealing_tanh(x, epoch) + tf.stop_gradient(tf.math.sign(x) - annealing_tanh(x, epoch))
    model = Model(inputs, x, name="encoder_unit")
    return model
def Encoder_module_annealing(L):
    def encoder_module(x, N):
        x = Dense(50)(x)
        x = LeakyReLU()(x)
        x = Dense(20)(x)
        x = LeakyReLU()(x)
        x = Dense(L)(x)
        # x = sign_relu_STE(x)
        x = annealing_tanh(x, N) + tf.stop_gradient(tf.sign(x) - annealing_tanh(x, N))
        # x = hard_tanh(x) + tf.stop_gradient(tf.sign(x) - hard_tanh(x))
        return x
    return encoder_module
def create_encoding_model_with_annealing_LSTM(k, l, input_shape):
    inputs = Input(shape=input_shape)
    epoch = inputs[:, 1][0]
    inputs_mod = inputs[:, 1:]
    x_list = tf.split(inputs_mod, num_or_size_splits=k, axis=1)
    encoding = Encoder_module_annealing(l)(x_list[0], epoch)
    for item in x_list[1:]:
        encoding = tf.concat((encoding, Encoder_module_annealing(l)(item, epoch)), axis=1)
    out = create_LSTM_model_with2states(k, [k, 1], 10)
    model = Model(inputs, out, name="k2_L2_annealing_nn_LSTM")
    print(model.summary())
    return model
def ensemble_regression(k, input_shape):
    regression_1 = create_regression_MLP_netowkr(input_shape, k)
    regression_2 = tf.keras.models.load_model("trained_models/Sept 19th/N_10000_5_Layer_MLP_regression.h5")
def create_uniform_encoding_model_with_annealing(k, l, input_shape):
    inputs = Input(shape=input_shape)
    x_list = tf.split(inputs, num_or_size_splits=k+1, axis=1)
    encoder_module = Uniform_Encoder_module_with_annealing(1, l, (2,))
    x_mod = tf.concat((x_list[0], x_list[1]), axis=1)
    encoding = encoder_module(x_mod)
    for item in x_list[2:]:
        x_mod = tf.concat((x_list[0], item), axis=1)
        encoding = tf.concat((encoding, encoder_module(x_mod)), axis=1)
    out = perception_model(encoding, k, 5)
    model = Model(inputs, out, name="auto_encoder_nn")
    return model
def perception_model(x, output, layer, logit=True):
    for i in range(layer-1):
        x = Dense(50)(x)
        x = LeakyReLU()(x)
    x = Dense(output)(x)
    if logit:
        return x
    else:
        return Softmax(x)

############################## Encoding models with bitstring input ##############################
def F_Encoder_module_annealing(L, i=0):
    def encoder_module(x, N):
        x = Dense(7, name="encoder_dense_1_{}".format(i))(x)
        x = LeakyReLU()(x)
        x = Dense(5, name="encoder_dense_4_{}".format(i))(x)
        x = LeakyReLU()(x)
        x = Dense(5, name="encoder_dense_2_{}".format(i))(x)
        x = LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = Dense(L, name="encoder_dense_3_{}".format(i))(x)
        # x = annealing_tanh(x, N, name="tanh_pos_{}".format(i)) + \
        #     tf.stop_gradient(tf.math.sign(x, name="encoder_sign_{}".format(i)) - annealing_tanh(x, N, name="tanh_neg_{}".format(i)))
        x = tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_pos_{}".format(i)) + tf.stop_gradient(binary_activation(x) - tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_neg_{}".format(i)))
        # x = annealing_tanh(tf.keras.layers.ReLU()(x), N, name="tanh_pos_{}".format(i)) + tf.stop_gradient(
        #     binary_activation(x) - annealing_tanh(tf.keras.layers.ReLU()(x), N, name="tanh_neg_{}".format(i)))
        return x
    return encoder_module
def F_create_encoding_model_with_annealing(k, l, input_shape):
    # features_mod = tf.ones((features.shape[1], 1)) * N
    # features_mod = tf.concat((features_mod, features), axis=2)
    # F_create_encoding_model_with_annealing(2, 1, (2, 24))
    inputs = Input(shape=input_shape)
    epoch = inputs[0, 0, 0]
    inputs_mod = inputs[:, :, 1:]
    x_list = tf.split(inputs_mod, num_or_size_splits=k, axis=1)
    encoding = F_Encoder_module_annealing(l, 0)(x_list[0][:, 0, :], epoch)
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, F_Encoder_module_annealing(l, i)(x_list[i][:, 0, :], epoch)), axis=1)
    x = tf.keras.layers.Dense(20)(encoding)
    x = sigmoid(x)
    out = tf.keras.layers.Dense(k)(x)
    model = Model(inputs, out, name="k2_L2_annealing_nn_on_floatpoint_bit")
    print(model.summary())
    return model
def F_create_LSTM_encoding_model_with_annealing(k, l, input_shape):
    # features_mod = tf.ones((features.shape[1], 1)) * N
    # features_mod = tf.concat((features_mod, features), axis=2)
    # F_create_encoding_model_with_annealing(2, 1, (2, 24))
    inputs = Input(shape=input_shape)
    epoch = inputs[0, 0, 0]
    inputs_mod = inputs[:, :, 1:]
    x_list = tf.split(inputs_mod, num_or_size_splits=k, axis=1)
    encoding = F_LSTM_Encoder_module_annealing(l, 0)(x_list[0][:, 0, :], epoch)
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, F_LSTM_Encoder_module_annealing(l, i)(x_list[i][:, 0, :], epoch)), axis=1)
    x = tf.keras.layers.Dense(20)(encoding)
    x = sigmoid(x)
    out = tf.keras.layers.Dense(2)(x)
    model = Model(inputs, out, name="k2_L2_annealing_nn_on_floatpoint_bit_CNN")
    print(model.summary())
    return model
def F_LSTM_Encoder_module_annealing(L, i=0):
    def encoder_module(x, N):
        x = tf.keras.layers.Reshape((23, 1))(x)
        x = tf.keras.layers.LSTM(4, name="LSTM_{}".format(i), kernel_initializer=tf.keras.initializers.he_uniform())(x)
        x = Dense(5, name="encoder_dense_1_{}".format(i), kernel_initializer=tf.keras.initializers.he_uniform())(x)
        x = LeakyReLU()(x)
        x = Dense(5, name="encoder_dense_2_{}".format(i), kernel_initializer=tf.keras.initializers.he_uniform())(x)
        x = LeakyReLU()(x)
        x = Dense(L, name="encoder_dense_3_{}".format(i), kernel_initializer=tf.keras.initializers.he_uniform())(x)
        # x = leaky_hard_tanh(tf.keras.layers.ReLU()(x)) + tf.stop_gradient(
        #     binary_activation(x) - leaky_hard_tanh(tf.keras.layers.ReLU()(x)))
        # x = annealing_tanh(x, N, name="tanh_pos_{}".format(i)) + \
        #     tf.stop_gradient(tf.math.sign(x, name="encoder_sign_{}".format(i)) - annealing_tanh(x, N, name="tanh_neg_{}".format(i)))
        x = tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_pos_{}".format(i)) + tf.stop_gradient(binary_activation(x) - tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_neg_{}".format(i)))
        # x = annealing_tanh(tf.keras.layers.ReLU()(x), N, name="tanh_pos_{}".format(i)) + tf.stop_gradient(
        #     binary_activation(x) - annealing_tanh(tf.keras.layers.ReLU()(x), N, name="tanh_neg_{}".format(i)))
        return x
    return encoder_module
def F_create_CNN_encoding_model_with_annealing(k, l, input_shape):
    # features_mod = tf.ones((features.shape[1], 1)) * N
    # features_mod = tf.concat((features_mod, features), axis=2)
    # F_create_encoding_model_with_annealing(2, 1, (2, 24))
    inputs = Input(shape=input_shape)
    epoch = inputs[0, 0, 0]
    inputs_mod = inputs[:, :, 1:]
    x_list = tf.split(inputs_mod, num_or_size_splits=k, axis=1)
    encoding = F_CNN_Encoder_module_annealing(l, 0)(x_list[0][:, 0, :], epoch)
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, F_CNN_Encoder_module_annealing(l, i)(x_list[i][:, 0, :], epoch)), axis=1)
    x = tf.keras.layers.Dense(20)(encoding)
    x = sigmoid(x)
    out = tf.keras.layers.Dense(2)(x)
    model = Model(inputs, out, name="k2_L2_annealing_nn_on_floatpoint_bit_CNN")
    print(model.summary())
    return model
def F_CNN_Encoder_module_annealing(L, i=0):
    def encoder_module(x, N):
        x = tf.keras.layers.Reshape((23, 1))(x)
        # x = tf.keras.layers.LSTM(8, name="LSTM_{}".format(i), kernel_initializer=tf.keras.initializers.he_normal())(x)
        x = tf.keras.layers.Conv1D(filters=10, kernel_size=5, strides=1)(x)
        x = tf.keras.layers.Reshape((70,))(x)
        x = Dense(5, name="encoder_dense_1_{}".format(i))(x)
        x = LeakyReLU()(x)
        x = Dense(5, name="encoder_dense_2_{}".format(i))(x)
        x = LeakyReLU()(x)
        x = Dense(5, name="encoder_dense_3_{}".format(i))(x)
        x = LeakyReLU()(x)
        x = Dense(L, name="encoder_dense_4_{}".format(i))(x)
        # x = annealing_tanh(x, N, name="tanh_pos_{}".format(i)) + \
        #     tf.stop_gradient(tf.math.sign(x, name="encoder_sign_{}".format(i)) - annealing_tanh(x, N, name="tanh_neg_{}".format(i)))
        x = tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_pos_{}".format(i)) + tf.stop_gradient(binary_activation(x) - tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_neg_{}".format(i)))
        # x = annealing_tanh(tf.keras.layers.ReLU()(x), N, name="tanh_pos_{}".format(i)) + tf.stop_gradient(
        #     binary_activation(x) - annealing_tanh(tf.keras.layers.ReLU()(x), N, name="tanh_neg_{}".format(i)))
        return x
    return encoder_module
def Thresholdin_network(input_shape):
    inputs = Input(shape=input_shape)
    epoch = inputs[0, 0, 0]
    inputs_mod = inputs[:, :, 1:]
    x = SubtractLayer_with_noise(name="threshold")(inputs_mod)
    x = tf.keras.layers.Reshape((2,))(x)
    x = tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_pos") + tf.stop_gradient(
        binary_activation(x) - tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_neg"))
    x = Dense(20, name="decoder_dense_1")(x)
    x = LeakyReLU()(x)
    x = Dense(2, name="decoder_dense_2")(x)
    model = Model(inputs, x)
    print(model.summary())
    print(model.trainable_variables)
    return model
def F_swapadoo_Encoder(k, l, input_shape):
    inputs = Input(shape=input_shape)
############################## Encoding models with Prof Yu's proposal input ##############################
def F_create_encoding_regression_module(input_shape, levels, j=0):
    inputs = Input(shape=input_shape)
    x = Dense(8, name = "encoding_dense_1_"+str(j))(inputs)
    x = LeakyReLU()(x)
    x = Dense(5, name = "encoding_dense_2_"+str(j))(x)
    x = LeakyReLU()(x)
    x = Dense(5, name = "encoding_dense_3_"+str(j))(x)
    x = LeakyReLU()(x)
    x = Dense(1, name = "encoding_output_real_"+str(j))(x)
    x = hard_sigmoid(x)
    x = hard_sigmoid(x) + tf.stop_gradient(multi_level_thresholding(x, levels) - hard_sigmoid(x))
    # x = annealing_tanh(x, epoch) + tf.stop_gradient(tf.math.sign(x) - annealing_tanh(x, epoch))
    model = Model(inputs, x, name="encoder_unit_"+str(j))
    return model
def F_creating_common_encoding_regression(input_shape, levels=2, k=2):
    inputs = Input(shape=input_shape)
    epoch = inputs[0, 0, 0]
    inputs_mod = inputs[:, :, 1:]
    x_list = tf.split(inputs_mod, num_or_size_splits=k, axis=1)
    encoding_model = F_create_encoding_regression_module((inputs_mod.shape[2]), levels)
    encoding = encoding_model(x_list[0][:, 0, :])
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, encoding_model(x_list[i][:, 0, :])), axis=1)
    model = Model(inputs, encoding, name="encoder_network")
    return model
def F_creating_distinct_encoding_regression(input_shape, levels=2, k=2):
    inputs = Input(shape=input_shape)
    epoch = inputs[0, 0, 0]
    inputs_mod = inputs[:, :, 1:]
    x_list = tf.split(inputs_mod, num_or_size_splits=k, axis=1)
    encoding = F_create_encoding_regression_module((inputs_mod.shape[2]), levels)(x_list[0][:, 0, :])
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding,  F_create_encoding_regression_module((inputs_mod.shape[2]), levels, j=i)(x_list[i][:, 0, :])), axis=1)
    model = Model(inputs, encoding, name="encoder_network")
    return model

############################## FDD Scheduling Models ##############################
def CommonFDD_Quantizer(M, B, K, i=0):
    inputs = Input(shape=[M, ])
    x = Dense(M*K)(inputs)
    x = LeakyReLU()(x)
    x = Dense(M*K)(x)
    x = LeakyReLU()(x)
    x = Dense(K)(x)
    x = LeakyReLU()(x)
    x = Dense(B)(x)
    x = tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_pos_{}".format(i)) + tf.stop_gradient(binary_activation(x) - tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_neg_{}".format(i)))
    model = Model(inputs, x, name="commonFDD_quantizer")
    return model
def FDD_encoding_model_no_constraint(M, K, B):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    x = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    quantizer = CommonFDD_Quantizer(2*M, B ,K)
    x_list = tf.split(x, num_or_size_splits=K, axis=1)
    reshaper = tf.keras.layers.Reshape((2*M,))
    encoding = quantizer(reshaper(x_list[0]))
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, quantizer(reshaper(x_list[i]))), axis=1)
    x = Dense(2*M*K)(encoding)
    x = LeakyReLU()(x)
    x = Dense(M*K)(x)
    # to be removed
    # output = tf.tanh(tf.keras.layers.ReLU()(x))
    output = sigmoid(x)
    model = Model(inputs, output)
    return model
def Floatbits_FDD_encoding_model_no_constraint(M, K, B):
    inputs = Input(shape=(K, M*2*23), dtype=tf.float32)
    quantizer = CommonFDD_Quantizer(M*2*23, B ,K)
    x_list = tf.split(inputs, num_or_size_splits=K, axis=1)
    reshaper = tf.keras.layers.Reshape((2*M*23,))
    encoding = quantizer(reshaper(x_list[0]))
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, quantizer(reshaper(x_list[i]))), axis=1)
    x = Dense(2*M*K)(encoding)
    x = LeakyReLU()(x)
    x = Dense(M*K)(x)
    # to be removed
    # output = tf.tanh(tf.keras.layers.ReLU()(x))
    output = sigmoid(x)
    model = Model(inputs, output)
    return model
def FDD_encoding_model_constraint_13_with_softmax(M, K, B):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    x = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    quantizer = CommonFDD_Quantizer(2*M, B ,K)
    x_list = tf.split(x, num_or_size_splits=K, axis=1)
    reshaper = tf.keras.layers.Reshape((2*M,))
    encoding = quantizer(reshaper(x_list[0]))
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, quantizer(reshaper(x_list[i]))), axis=1)
    x = Dense(M**2)(encoding)
    x = LeakyReLU()(x)
    x = Dense(M*K)(encoding)
    x = LeakyReLU()(x)
    x = Dense(M*K)(x)
    # to be removed
    x_list2 = tf.split(x, num_or_size_splits=K, axis=1)
    output = tf.keras.layers.Softmax()(x_list2[0])
    for i in range(1, len(x_list2)):
        output = tf.concat((output, tf.keras.layers.Softmax()(x_list2[i])), axis=1)
    # yep
    # x = tf.sigmoid(x)
    model = Model(inputs, output)
    return model
def Floatbits_FDD_encoding_model_constraint_13_with_softmax(M, K, B):
    inputs = Input(shape=(K, M * 2 * 23), dtype=tf.float32)
    quantizer = CommonFDD_Quantizer(M * 2 * 23, B, K)
    x_list = tf.split(inputs, num_or_size_splits=K, axis=1)
    reshaper = tf.keras.layers.Reshape((2 * M * 23,))
    encoding = quantizer(reshaper(x_list[0]))
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, quantizer(reshaper(x_list[i]))), axis=1)
    x = Dense(M**2)(encoding)
    x = LeakyReLU()(x)
    x = Dense(M*K)(encoding)
    x = LeakyReLU()(x)
    x = Dense(M*K)(x)
    # to be removed
    x_list2 = tf.split(x, num_or_size_splits=K, axis=1)
    output = tf.keras.layers.Softmax()(x_list2[0])
    for i in range(1, len(x_list2)):
        output = tf.concat((output, tf.keras.layers.Softmax()(x_list2[i])), axis=1)
    # yep
    # x = tf.sigmoid(x)
    model = Model(inputs, output)
    return model
def LSTM_Ranking_model(M, K, k):
    inputs = Input(shape=[M*K,], name="ranking_network_input")
    x_reshape = tf.expand_dims(inputs, 1)
    x = tf.tile(x_reshape, [1, k, 1])
    lstm_ranker = tf.keras.layers.LSTM(K, return_sequences=True)
    x = lstm_ranker(x)
    x = tf.nn.softmax(x, axis=2)
    x = tf.reduce_sum(x, axis=1)
    model = Model(inputs, x)
    return model
def FDD_encoding_model_constraint_123_with_softmax_and_soft_mask(M, K, B, k=3):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    x = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    quantizer = CommonFDD_Quantizer(2*M, B ,K)
    x_list = tf.split(x, num_or_size_splits=K, axis=1)
    reshaper = tf.keras.layers.Reshape((2*M,))
    encoding = quantizer(reshaper(x_list[0]))
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, quantizer(reshaper(x_list[i]))), axis=1)
    x = Dense(2*M*K, name="start_of_decoding")(encoding)
    x = LeakyReLU()(x)
    x = Dense(2*M*K)(x)
    x = LeakyReLU()(x)
    x = Dense(M*K)(x)
    ranking_output = LSTM_Ranking_model(M, K, k)(x)
    # ranking_output = Dense(K*M)(x)
    # ranking_output = Dense(K)(ranking_output)
    # ranking_output = tf.tanh(LeakyReLU()(ranking_output))
    # perform softmax to all the results
    x_list2 = tf.split(x, num_or_size_splits=K, axis=1)
    output = tf.keras.layers.Softmax()(x_list2[0])
    for i in range(1, len(x_list2)):
        output = tf.concat((output, tf.keras.layers.Softmax()(x_list2[i])), axis=1)
    # yep
    # x = tf.sigmoid(x)
    output = tf.concat((output, ranking_output), axis=1)
    model = Model(inputs, output)
    return model
def Floatbits_FDD_encoding_model_constraint_123_with_softmax_and_soft_mask(M, K, B, k=3):
    inputs = Input(shape=(K, M*2*23), dtype=tf.float32)
    quantizer = CommonFDD_Quantizer(M*2*23, B ,K)
    x_list = tf.split(inputs, num_or_size_splits=K, axis=1)
    reshaper = tf.keras.layers.Reshape((2*M*23,))
    encoding = quantizer(reshaper(x_list[0]))
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, quantizer(reshaper(x_list[i]))), axis=1)
    x = Dense(2*M*K, name="start_of_decoding")(encoding)
    x = LeakyReLU()(x)
    x = Dense(2*M*K)(x)
    x = LeakyReLU()(x)
    x = Dense(M*K)(x)
    ranking_output = LSTM_Ranking_model(M, K, k)(x)
    # ranking_output = Dense(K*M)(x)
    # ranking_output = Dense(K)(ranking_output)
    # ranking_output = tf.tanh(LeakyReLU()(ranking_output))
    # perform softmax to all the results
    x_list2 = tf.split(x, num_or_size_splits=K, axis=1)
    output = tf.keras.layers.Softmax()(x_list2[0])
    for i in range(1, len(x_list2)):
        output = tf.concat((output, tf.keras.layers.Softmax()(x_list2[i])), axis=1)
    # yep
    # x = tf.sigmoid(x)
    output = tf.concat((output, ranking_output), axis=1)
    model = Model(inputs, output)
    return model
def FDD_encoding_model_constraint_13_with_regularization(M, K, B):
    inputs = Input(shape=(K, M), dtype=tf.complex128)
    x = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    quantizer = CommonFDD_Quantizer(2*M, B ,K)
    x_list = tf.split(x, num_or_size_splits=K, axis=1)
    reshaper = tf.keras.layers.Reshape((2*M,))
    encoding = quantizer(reshaper(x_list[0]))
    for i in range(1, len(x_list)):
        encoding = tf.concat((encoding, quantizer(reshaper(x_list[i]))), axis=1)
    x = Dense(2*M*K)(encoding)
    x = LeakyReLU()(x)
    x = Dense(M*K)(x)
    x = tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_neg_output") + tf.stop_gradient(binary_activation(x) - tf.tanh(tf.keras.layers.ReLU()(x), name="tanh_neg_output"))
    model = Model(inputs, x)
    return model

############################## FDD Scheduling Models No quantizing ##############################
def FDD_model_no_constraint(M, K, B):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    x = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    # create input vector
    reshaper = tf.keras.layers.Reshape((2 * M * K,))
    x = reshaper(x)
    x = Dense(M * M)(x)
    x = LeakyReLU()(x)
    x = Dense(M * M)(x)
    x = LeakyReLU()(x)
    x = Dense(M * M)(x)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    # to be removed
    output = tf.tanh(tf.keras.layers.ReLU()(x))
    # output = sigmoid(x)
    model = Model(inputs, output)
    return model
def Floatbits_FDD_model_no_constraint(M, K, B):
    inputs = Input(shape=(K, M * 2 * 23), dtype=tf.float32)
    # create input vector
    reshaper = tf.keras.layers.Reshape((2 * M * K * 23,))
    x = reshaper(inputs)
    x = Dense(M * M)(x)
    x = LeakyReLU()(x)
    x = Dense(M * M)(x)
    x = LeakyReLU()(x)
    x = Dense(M * M)(x)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    # to be removed
    output = tf.tanh(tf.keras.layers.ReLU()(x))
    # output = sigmoid(x)
    model = Model(inputs, output)
    return model
def FDD_model_softmax(M, K, B):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    x = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    # create input vector
    reshaper = tf.keras.layers.Reshape((2 * M * K,))
    x = reshaper(x)
    x = Dense(M * M)(x)
    x = LeakyReLU()(x)
    x = Dense(M * M)(x)
    x = LeakyReLU()(x)
    x = Dense(M * M)(x)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    # to be removed
    x_list2 = tf.split(x, num_or_size_splits=K, axis=1)
    output = tf.keras.layers.Softmax()(x_list2[0])
    for i in range(1, len(x_list2)):
        output = tf.concat((output, tf.keras.layers.Softmax()(x_list2[i])), axis=1)
    # yep
    # x = tf.sigmoid(x)
    model = Model(inputs, output)
    return model

if __name__ == "__main__":
    # F_create_encoding_model_with_annealing(2, 1, (2, 24))
    # F_create_CNN_encoding_model_with_annealing(2, 1, (2, 24))
    # print(Thresholdin_network((2, )).summary())
    M = 20
    K = 5
    B = 5
    model = FDD_encoding_model_constraint_123_with_softmax_and_ranking(M, K, B)
