import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Softmax, Input, ThresholdedReLU, Flatten
from tensorflow.keras.activations import sigmoid
import random
# ============================  Data gen ============================

def gen_data(N, k, low=0, high=1, batchsize=30):
    channel_data = tf.random.uniform((N,k), low, high)
    channel_label = tf.math.argmax(channel_data, axis=1)
    dataset = Dataset.from_tensor_slices((channel_data, channel_label)).batch(batchsize)
    return dataset

# ============================  Metrics  ============================
class ExpectedThroughput(tf.keras.metrics.Metric):
    def __init__(self, name='expected_throughput', **kwargs):
        super(ExpectedThroughput, self).__init__(name=name, **kwargs)
        self.max_throughput = self.add_weight(name='tp', initializer='zeros')
        self.expected_throughput = self.add_weight(name='tp2', initializer='zeros')
        self.count = self.add_weight(name='tp3', initializer='zeros')
        self.a = tf.math.log(tf.cast(2, dtype=tf.float32))

    def update_state(self, y_true, y_pred, x):
        c_max = tf.gather(x, y_true, axis=1, batch_dims=1)
        i_max = tf.math.log(1 - 1.5*tf.math.log(1 - c_max))/self.a
        weighted_c = tf.math.multiply(Softmax()(y_pred), x)
        i_expected = tf.math.reduce_sum(tf.math.log(1 - 1.5*tf.math.log(1 - weighted_c))/self.a, axis=1)
        self.max_throughput.assign_add(tf.math.reduce_sum(i_max, axis=0))
        self.expected_throughput.assign_add(tf.math.reduce_sum(i_expected, axis=0))
        self.count.assign_add(y_true.shape[0])
    def result(self):
        return self.max_throughput/self.count, self.expected_throughput/self.count
# ============================  Loss fn  ============================

def Throughout_diff_Loss():
    def through_put_loss(y_true, y_pred, x=None):
        a = tf.math.log(tf.cast(2, dtype=tf.float32))
        c_max = tf.gather(x, y_true, axis=1, batch_dims=1)
        i_max = tf.math.log(1 - 1.5 * tf.math.log(1 - c_max)) / a
        weighted_c = tf.math.multiply(Softmax()(y_pred), x)
        i_expected = tf.math.reduce_sum(tf.math.log(1 - 1.5 * tf.math.log(1 - weighted_c)) / a, axis=1)
        return tf.square(tf.math.reduce_sum(i_max, axis=0) - tf.math.reduce_sum(i_expected, axis=0))
    return through_put_loss
def Throughput():
    def through_put_loss(y_pred, x=None):
        # apply softmax to get distribution
        a = tf.math.log(tf.cast(2, dtype=tf.float32))
        weighted_c = tf.math.multiply(Softmax()(y_pred), x)
        i_expected = tf.math.reduce_sum(tf.math.log(1 - 1.5 * tf.math.log(1 - weighted_c)) / a, axis=1)
        cost = -tf.math.reduce_sum(i_expected, axis=0)
        return cost
    return through_put_loss
def Mix_loss():
    def mixloss(y_true, y_pred, x=None):
        a = tf.math.log(tf.cast(2, dtype=tf.float32))
        c_max = tf.gather(x, y_true, axis=1, batch_dims=1)
        i_max = tf.math.log(1 - 1.5 * tf.math.log(1 - c_max)) / a
        weighted_c = tf.math.multiply(Softmax()(y_pred), x)
        i_expected = tf.math.reduce_sum(tf.math.log(1 - 1.5 * tf.math.log(1 - weighted_c)) / a, axis=1)
        loss1 = tf.square(tf.math.reduce_sum(i_max, axis=0) - tf.math.reduce_sum(i_expected, axis=0))
        loss2 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred)
        return loss2 + loss1/100
    return mixloss