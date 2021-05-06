from models import *
from util import *
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns

if __name__ == "__main__":
    num_data = 20
    K = 20
    M = 64
    N_rf = 4
    ds = tf.abs(gen_realistic_data("trained_models/Apr5th/K20/user_positions.npy", num_data, K, M, N_rf))
    max = tf.reduce_max(tf.reduce_max(ds, axis=2, keepdims=True), axis=1, keepdims=True)
    ds = ds / max
    layer = Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum(K, M, N_rf, 10)
    raw_out_put_0 = tf.stop_gradient(tf.multiply(tf.zeros((K, M)), ds[:, :, :]) + 1.0)
    raw_out_put_0 = tf.tile(tf.expand_dims(raw_out_put_0, axis=3), (1, 1, 1, N_rf))
    raw_out_put_0 = tf.keras.layers.Reshape((K * M, N_rf))(raw_out_put_0)
    input_i = layer(raw_out_put_0, ds, 1 - 1.0)
    i1 = tf.reshape(input_i, [20*1280, 21])
    ax = sns.boxplot(data=i1)
    plt.show()