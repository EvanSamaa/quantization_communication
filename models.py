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
############################## Layers ##############################
class Closest_embedding_layer(tf.keras.layers.Layer):
    def __init__(self, user_count=2, embedding_count=8, bit_count=15, i=0, **kwargs):
        super(Closest_embedding_layer, self).__init__()
        self.user_count = user_count
        self.bit_count = bit_count
        self.embedding_count = embedding_count
        initializer = tf.keras.initializers.he_normal()
        # self.E = tf.Variable(initializer(shape=[self.embedding_count, self.bit_count]), trainable=True)
        self.E = tf.Variable(tf.random.normal(([self.embedding_count, self.bit_count]), 0, 0.01)
        , trainable=True)
        self.i = i
    def call(self, z, training=True):
        # if training:
        #     z = z + tf.random.normal((z.shape[1], z.shape[2]), 0, 0.01)
        z = tf.expand_dims(z, 2)
        z = tf.tile(z, (1, 1, self.E.shape[0], 1))
        eu_distance = tf.reduce_sum(tf.square(tf.stop_gradient(z) - self.E), axis=3) # calculate euclidience distance between
        k = tf.argmin(eu_distance, axis=2)
        output = tf.gather(self.E, k)
        return output
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'bit_count':self.bit_count,
            'user_count':self.user_count,
            'embedding_count':self.embedding_count,
            'i':self.i,
            'name':"Closest_embedding_layer_{}".format(self.i)
        })
        return config

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

def Autoencoder_Encoding_module(k, l, input_shape, i=0, code_size=15):
    inputs = Input(input_shape, dtype=tf.float32)
    x = Dense(64, kernel_initializer=tf.keras.initializers.he_normal())(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = Dense(code_size, kernel_initializer=tf.keras.initializers.he_normal())(x)
    x = tf.keras.layers.Reshape((1, code_size))(x)
    return Model(inputs, x, name="encoder_{}".format(i))
def Autoencoder_Decoding_module(k, l, input_shape):
    inputs = Input(input_shape)
    x = Dense(64, kernel_initializer=tf.keras.initializers.he_normal())(inputs)
    x = LeakyReLU()(x)
    x = Dense(k, kernel_initializer=tf.keras.initializers.he_normal())(x)
    return Model(inputs, x, name="decoder")
def DiscreteVAE(k, l, input_shape, code_size=15):
    inputs = Input(input_shape, dtype=tf.float32)
    x_list = tf.split(inputs, num_or_size_splits=k, axis=1)
    # list of modules
    encoder = Autoencoder_Encoding_module(k, l, (1, ), code_size=code_size)
    decoder = Autoencoder_Decoding_module(k, code_size, (k*code_size, ))
    find_nearest_e = Closest_embedding_layer(user_count=k, embedding_count=2**l, bit_count=code_size, i=0)
    encoding_reshaper = tf.keras.layers.Reshape((k*code_size, ), name="encoding_reshaper")
    # computation of encoding
    # z_e = Autoencoder_Encoding_module(k, l, (1, ))(x_list[0])
    z_e = encoder(x_list[0])
    i = 1
    for item in x_list[1:]:
        # z_e = tf.concat((z_e, Autoencoder_Encoding_module(k, l, (1, ), i)(item)), axis=1)
        z_e = tf.concat((z_e, encoder(item)), axis=1)
        i = i + 1
    z_qq = find_nearest_e(z_e)
    z_fed_forward = z_e + tf.stop_gradient(z_qq - z_e)
    z_fed_forward = encoding_reshaper(z_fed_forward)
    z_e = encoding_reshaper(z_e)
    z_qq = encoding_reshaper(z_qq)
    out = decoder(z_fed_forward)
    output_all = tf.keras.layers.concatenate((out, z_qq, z_e), 1)
    model = Model(inputs, output_all, name="DiscreteVAE")
    return model
def DiscreteVAE_diff_scheduler(k, l, input_shape, code_size=15):
    inputs = Input(input_shape, dtype=tf.float32)
    x_list = tf.split(inputs, num_or_size_splits=k, axis=1)
    # list of modules
    decoder = Autoencoder_Decoding_module(k, code_size, (k*code_size, ))
    encoding_reshaper = tf.keras.layers.Reshape((k*code_size, ), name="encoding_reshaper")
    # computation of encoding
    # z_e = Autoencoder_Encoding_module(k, l, (1, ))(x_list[0])
    z_e = Autoencoder_Encoding_module(k, l, (1, ), code_size=code_size)(x_list[0])
    z_q = Closest_embedding_layer(user_count=k, embedding_count=2**l, bit_count=code_size)(z_e)
    z_fed_forward = z_e + tf.stop_gradient(z_q - z_e)
    i = 1
    for item in x_list[1:]:
        # z_e = tf.concat((z_e, Autoencoder_Encoding_module(k, l, (1, ), i)(item)), axis=1)
        z_e_temp = Autoencoder_Encoding_module(k, l, (1, ), i = i, code_size=code_size)(item)
        z_q_temp = Closest_embedding_layer(user_count=k, embedding_count=2**l, bit_count=code_size)(z_e_temp)
        z_fed_forward_temp = z_e_temp + tf.stop_gradient(z_q_temp - z_e_temp)
        z_e = tf.concat((z_e, z_e_temp), axis=1)
        z_q = tf.concat((z_q, z_q_temp), axis=1)
        z_fed_forward = tf.concat((z_fed_forward, z_fed_forward_temp), axis=1)
        i = i + 1
    z_fed_forward = encoding_reshaper(z_fed_forward)
    z_e = encoding_reshaper(z_e)
    z_q = encoding_reshaper(z_q)
    out = decoder(z_fed_forward)
    output_all = tf.keras.layers.concatenate((out, z_q, z_e), 1)
    model = Model(inputs, output_all, name="DiscreteVAE")
    return model
def DiscreteVAE_regression(l, input_shape, code_size = 15):

    inputs = Input(input_shape, dtype=tf.float32)
    encoder = Autoencoder_Encoding_module(3, l, (1,), code_size=code_size)
    decoder = Autoencoder_Decoding_module(1, code_size, (code_size,))
    find_nearest_e = Closest_embedding_layer(1, 2**l, code_size)
    encoding_reshaper = tf.keras.layers.Reshape((code_size,), name="encoding_reshaper")
    z_e = encoder(inputs)
    z_qq = find_nearest_e(z_e)
    z_fed_forward = z_e + tf.stop_gradient(z_qq - z_e)
    z_fed_forward = encoding_reshaper(z_fed_forward)
    z_e = encoding_reshaper(z_e)
    z_qq = encoding_reshaper(z_qq)
    out = decoder(z_fed_forward)
    output_all = tf.keras.layers.concatenate((out, z_qq, z_e), 1)
    model = Model(inputs, output_all, name="DiscreteVAEregression")
    print(model.summary())
    return model

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
def LSTM_Ranking_model(M, K, k, sum_all=True):
    inputs = Input(shape=[M*K,], name="ranking_network_input")
    x_reshape = tf.expand_dims(inputs, 1)
    x = tf.tile(x_reshape, [1, k, 1])
    lstm_ranker = tf.keras.layers.LSTM(K, return_sequences=True)
    x = lstm_ranker(x)
    x = tf.nn.softmax(x, axis=2)
    if sum_all:
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
    x = Dense(3*M, kernel_initializer=tf.keras.initializers.he_normal())(x)
    x = LeakyReLU()(x)
    x = Dense(M, kernel_initializer=tf.keras.initializers.he_normal())(x)
    x = LeakyReLU()(x)
    x = Dense(M * K, kernel_initializer=tf.keras.initializers.he_normal())(x)
    # to be removed
    # output = tf.tanh(tf.keras.layers.ReLU()(x))
    # output = leaky_hard_sigmoid(x)
    output = sigmoid(x)
    model = Model(inputs, output)
    return model
# def Fully_connected_Ranking_Model(M, K, k):
#     inputs = Input(shape=(M*K), dtype=tf.float32)
#     x = Dense(M*K)
#     x = Dense(M*K)
def DNN_Ranking_NN_submodule(M, K, k, inputshape):
    inputs = Input(inputshape)
    x = Dense(M*K)(inputs)
    x = Dense(2*M*K)(x)
    x = Dense(M*K)(x)
    x = Dense(K)(x)
    x = tf.keras.layers.Softmax()(x)
    dnn_model = Model(inputs, x, name="dnn_softmax_network")
    return dnn_model
def DNN_Ranking_model(M, K, k, sum_all = False):
    inputs = Input((M*K, ))
    dnn = DNN_Ranking_NN_submodule(M, K, k, (M*K))
    stretch_matrix = np.zeros((M * K, K))
    for i in range(0, K):
        for j in range(0, M):
            stretch_matrix[i * M + j, i] = 1
    stretch_matrix = tf.constant(stretch_matrix, tf.float32)
    output = dnn(inputs)
    for i in range(1, k):
        # tiled_stretch_matrix = tf.tile(tf.expand_dims(stretch_matrix, 0), [1, 1, 1])
        stretched_rank_matrix = tf.matmul(stretch_matrix, tf.keras.layers.Reshape((output.shape[1], 1))(output))
        stretched_rank_matrix = tf.keras.layers.Reshape((stretched_rank_matrix.shape[1], ))(stretched_rank_matrix)
        output = output + dnn(tf.multiply(1-stretched_rank_matrix, inputs))
    model = Model(inputs, output, name="dnn_ranking_module")
    return model
def FDD_with_CNN(M, K, N_rf):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    c = tf.keras.layers.Conv2D(M, (1, 3))(tf.keras.layers.Reshape((K, 2*M, 1))(input_mod))
    c = tf.keras.layers.Reshape((c.shape[1]*c.shape[2]*c.shape[3], ))(c)
    x = tf.keras.layers.Reshape((2*K*M, ))(input_mod)
    x = tf.concat((c, x), axis=1)
    x = Dense(3*M*K)(x)
    x = LeakyReLU()(x)
    x = Dense(2*M*K)(x)
    x = LeakyReLU()(x)
    x = Dense(M*K)(x)
    x = tf.keras.layers.Softmax()(x)
    model = Model(inputs, x)
    print(model.summary())
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
    x = Dense(M * K)(x)
    # to be removed
    output = tf.tanh(tf.keras.layers.ReLU()(x))
    # output = tf.keras.layers.Softmax()(x)
    # output = sigmoid(x)
    model = Model(inputs, output)
    return model
def DNN_3_layer_model(input_shape, M, K, i=0):
    inputs = Input(shape=input_shape, dtype=tf.float32)
    x = Dense(3*M)(inputs)
    x = LeakyReLU()(x)
    x = Dense(M)(x)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    x = tf.keras.layers.Softmax()(x)
    model = Model(inputs, x, name="pass_{}".format(i))
    return model
def DNN_3_layer_Thicc_model(input_shape, M, K, i=0):
    inputs = Input(shape=input_shape, dtype=tf.float32)
    x = Dense(3*M*K)(inputs)
    x = LeakyReLU()(x)
    x = Dense(M*K)(x)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    x = tf.keras.layers.Softmax()(x)
    model = Model(inputs, x, name="pass_{}".format(i))
    return model
def FDD_softmax_k_times_with_magnitude(M, K, k):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.abs(inputs)
    input_mod = tf.keras.layers.Reshape((K * M,))(input_mod)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K*M)), input_mod[:, :K*M]))
    input_pass_0 = tf.keras.layers.Concatenate(axis=1)((decision_0, input_mod))
    # dnn_model = DNN_3_layer_model((3*K*M), M, K, 0)
    x = DNN_3_layer_Thicc_model((2*K*M), M, K, 0)(input_pass_0)
    for i in range(1, k):
        decision_i = x
        input_pass_i = tf.keras.layers.Concatenate(axis=1)((decision_i, input_mod))
        x = x + DNN_3_layer_Thicc_model((2*K*M), M, K, i)(input_pass_i)
    model = Model(inputs, x)
    return model
def FDD_softmax_k_times(M, K, k):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    input_mod = tf.keras.layers.Reshape((2 * K * M,))(input_mod)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K*M)), input_mod[:, :K*M]))
    input_pass_0 = tf.keras.layers.Concatenate(axis=1)((decision_0, input_mod))
    # dnn_model = DNN_3_layer_model((3*K*M), M, K, 0)
    x = DNN_3_layer_Thicc_model((3*K*M), M, K, 0)(input_pass_0)
    for i in range(1, k):
        decision_i = x
        input_pass_i = tf.keras.layers.Concatenate(axis=1)((decision_i, input_mod))
        x = x + DNN_3_layer_Thicc_model((3*K*M), M, K, i)(input_pass_i)
    model = Model(inputs, x)
    return model
def FDD_softmax_k_times_hard_output(M, K, k):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    input_mod = tf.keras.layers.Reshape((2 * K * M,))(input_mod)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K*M)), input_mod[:, :K*M]))
    input_pass_0 = tf.keras.layers.Concatenate(axis=1)((decision_0, input_mod))
    # dnn_model = DNN_3_layer_model((3*K*M), M, K, 0)
    x = DNN_3_layer_Thicc_model((3*K*M), M, K, 0)(input_pass_0)
    for i in range(1, k):
        decision_i = x + tf.stop_gradient(binary_activation(x, 0.5) - x)
        input_pass_i = tf.keras.layers.Concatenate(axis=1)((decision_i, input_mod))
        output_i = DNN_3_layer_model((3*K*M), M, K, i)(input_pass_i)
        x = x + output_i
    model = Model(inputs, x)
    return model
def FDD_softmax_k_times_hard_output_with_magnitude(M, K, k):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.abs(inputs)
    input_mod = tf.keras.layers.Reshape((K * M,))(input_mod)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K*M)), input_mod[:, :K*M]))
    input_pass_0 = tf.keras.layers.Concatenate(axis=1)((decision_0, input_mod))
    # dnn_model = DNN_3_layer_model((3*K*M), M, K, 0)
    x = DNN_3_layer_Thicc_model((2*K*M), M, K, 0)(input_pass_0)
    for i in range(1, k):
        decision_i = x + tf.stop_gradient(binary_activation(x, 0.5) - x)
        input_pass_i = tf.keras.layers.Concatenate(axis=1)((decision_i, input_mod))
        output_i = DNN_3_layer_model((2*K*M), M, K, i)(input_pass_i)
        x = x + output_i
    x = x + tf.stop_gradient(binary_activation(x, 0.5) - x)
    model = Model(inputs, x)
    return model
def FDD_softmax_k_times_common_dnn(M, K, k):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    input_mod = tf.keras.layers.Reshape((2 * K * M,))(input_mod)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K*M)), input_mod[:, :K*M]))
    input_pass_0 = tf.keras.layers.Concatenate(axis=1)((decision_0, input_mod))
    dnn_model = DNN_3_layer_model((3*K*M), M, K, 0)
    x = dnn_model(input_pass_0)
    for i in range(1, k):
        decision_i = x + tf.stop_gradient(binary_activation(x) - x)
        input_pass_i = tf.keras.layers.Concatenate(axis=1)((decision_i, input_mod))
        x = x + dnn_model(input_pass_i)
    model = Model(inputs, x)
    return model
def FDD_ranked_softmax(M, K, k):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    input_mod = tf.abs(inputs)
    input_mod = tf.keras.layers.Reshape((K * M,))(input_mod)
    decision_0 = tf.stop_gradient(tf.multiply(tf.zeros((K * M)), input_mod[:, :K * M]))
    input_pass_0 = tf.keras.layers.Concatenate(axis=1)((decision_0, input_mod))
    decision_1 = DNN_3_layer_Thicc_model((2*K*M), M, K, 0)(input_pass_0)
    output = tf.keras.layers.Reshape((M*K, 1))(decision_1)
    for i in range(1, k):
        input_pass_i = tf.keras.layers.Concatenate(axis=1)((decision_1, input_mod))
        output_i = DNN_3_layer_model((2 * K * M), M, K, i)(input_pass_i)
        x = x + output_i
        output = tf.concat((output, x), axis=2)
    model = Model(inputs, output)
    return model

def FDD_model_softmax(M, K, B):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    x = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    # create input vector
    reshaper = tf.keras.layers.Reshape((2 * M * K,))
    x = reshaper(x)
    x = Dense(3*M)(x)
    x = LeakyReLU()(x)
    x = Dense(M)(x)
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
def FDD_softmax_with_soft_mask(M, K, B, k=3):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    x = tf.keras.layers.Concatenate(axis=2)([tf.math.real(inputs), tf.math.imag(inputs)])
    # create input vector
    reshaper = tf.keras.layers.Reshape((2 * M * K,))
    x = reshaper(x)
    # normalize
    mean = tf.expand_dims(tf.reduce_mean(x, axis=1), 1)
    std = tf.expand_dims(tf.math.reduce_std(x, axis=1), 1)
    x = (x - mean)/std
    x = tf.keras.layers.Reshape((x.shape[1],))(x)
    # model starts
    x = Dense(M)(x)
    x = LeakyReLU()(x)
    x = Dense(M)(x)
    x = LeakyReLU()(x)
    x = Dense(M*K)(x)
    ranking_output = LSTM_Ranking_model(M, K, k)(x)
    x_list2 = tf.split(x, num_or_size_splits=K, axis=1)
    output = tf.keras.layers.Softmax()(x_list2[0])
    for i in range(1, len(x_list2)):
        output = tf.concat((output, tf.keras.layers.Softmax()(x_list2[i])), axis=1)
    # yep
    # x = tf.sigmoid(x)
    output = tf.concat((output, ranking_output), axis=1)
    model = Model(inputs, output)
    print(model.summary())
    return model
def FDD_softmax_with_unconstraint_soft_masks(M, K, B, k=3):
    inputs = Input(shape=(K, M), dtype=tf.complex64)
    mod_input = tf.abs(inputs)
    # create input vector
    reshaper = tf.keras.layers.Reshape((M * K,))
    mod_input = reshaper(mod_input)
    # normalize
    # mean = tf.expand_dims(tf.reduce_mean(mod_input, axis=1), 1)
    # std = tf.expand_dims(tf.math.reduce_std(mod_input, axis=1), 1)
    # x = (mod_input - mean)/std
    # x = tf.keras.layers.Reshape((x.shape[1],))(x)
    # model starts
    x = Dense(3*M)(mod_input)
    x = LeakyReLU()(x)
    x = Dense(M)(x)
    x = LeakyReLU()(x)
    x = Dense(M*K)(x)
    # ranking_output = Dense(50)(x)
    # ranking_output = Dense(20)(ranking_output)
    # ranking_output = Dense(k)(ranking_output)
    x_list2 = tf.split(x, num_or_size_splits=K, axis=1)
    output = tf.keras.layers.Softmax()(x_list2[0])
    for i in range(1, len(x_list2)):
        output = tf.concat((output, tf.keras.layers.Softmax()(x_list2[i])), axis=1)
    # yep
    # x = tf.sigmoid(x)
    # ranking_output = tf.keras.layers.Reshape((k*K,))(ranking_output)
    ranking_input = tf.concat((mod_input, x), axis=1)
    ranking_output = Dense(M * K)(ranking_input)
    ranking_output = LeakyReLU()(ranking_output)
    ranking_output = Dense(M)(ranking_output)
    ranking_output = LeakyReLU()(ranking_output)
    ranking_output = Dense(K)(ranking_output)
    ranking_output = sigmoid(ranking_output)
    # ranking_output = tf.tanh(tf.keras.layers.ReLU()(ranking_output))

    output = tf.concat((output, ranking_output), axis=1)
    model = Model(inputs, output)
    return model

def Floatbits_FDD_model_softmax(M, K, B):
    inputs = Input(shape=(K, M * 2 * 23), dtype=tf.float32)
    # create input vector
    reshaper = tf.keras.layers.Reshape((2 * M * K * 23,))
    x = reshaper(inputs)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    x = LeakyReLU()(x)
    x = Dense(M * M)(x)
    x = LeakyReLU()(x)
    x = Dense(M * K)(x)
    x_list2 = tf.split(x, num_or_size_splits=K, axis=1)
    output = tf.keras.layers.Softmax()(x_list2[0])
    for i in range(1, len(x_list2)):
        output = tf.concat((output, tf.keras.layers.Softmax()(x_list2[i])), axis=1)
    model = Model(inputs, output)
    return model

if __name__ == "__main__":
    # F_create_encoding_model_with_annealing(2, 1, (2, 24))
    # F_create_CNN_encoding_model_with_annealing(2, 1, (2, 24))
    # print(Thresholdin_network((2, )).summary())
    # DiscreteVAE(2, 4, (2,))
    N = 1000
    M = 20
    K = 10
    B = 10
    seed = 200
    N_rf = 3
    model = FDD_with_CNN(M, K, N_rf)
