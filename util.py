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
def gen_number_data(N=10000, k = 7.5, batchsize=10000):
    channel_data_num = tf.random.uniform((N, 1), 0, k)
    channel_data_num = tf.cast(tf.round(channel_data_num), dtype=tf.int32)
    # channel_data_num = tf.round(channel_data_num)
    channel_data = tf.cast(tf.one_hot(channel_data_num, depth=8, on_value=1.0, off_value=0.0), tf.float32)
    channel_data = tf.reshape(channel_data, (N, 8))
    channel_label = channel_data_num
    dataset = Dataset.from_tensor_slices((channel_data, channel_label)).batch(batchsize)
    return dataset
def gen_encoding_data(N=1000, Sequence_length=10000, k=8, batchsize = 100):
    dict_list = [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]
    output = np.zeros((N, Sequence_length, 3))
    channel_label = np.zeros((N, 1))
    for n in range(N):
        random.shuffle(dict_list)
        current = random.randint(1,8)
        current_encoding = np.array(dict_list[:current])
        idx = np.random.randint(0, current, size=Sequence_length)
        for m in range(Sequence_length):
            output[n, m] = current_encoding[idx[m]]
        channel_label[n] = current
    channel_label = tf.cast(channel_label, tf.float32)
    output = tf.cast(output, tf.float32)
    dataset = Dataset.from_tensor_slices((output, channel_label)).batch(batchsize)
    return dataset

# ============================  Metrics  ============================
class ExpectedThroughput(tf.keras.metrics.Metric):
    def __init__(self, name='expected_throughput', logit=True, **kwargs):
        super(ExpectedThroughput, self).__init__(name=name, **kwargs)
        self.max_throughput = self.add_weight(name='tp', initializer='zeros')
        self.expected_throughput = self.add_weight(name='tp2', initializer='zeros')
        self.count = self.add_weight(name='tp3', initializer='zeros')
        self.a = tf.math.log(tf.cast(2, dtype=tf.float32))
        self.logit = logit

    def update_state(self, y_true, y_pred, x):
        c_max = tf.gather(x, y_true, axis=1, batch_dims=1)
        i_max = tf.math.log(1 - 1.5*tf.math.log(1 - c_max))/self.a
        if self.logit:
            weighted_c = tf.math.multiply(Softmax()(y_pred), x)
        else:
            weighted_c = tf.math.multiply(y_pred, x)
        i_expected = tf.math.reduce_sum(tf.math.log(1 - 1.5*tf.math.log(1 - weighted_c))/self.a, axis=1)
        self.max_throughput.assign_add(tf.math.reduce_sum(i_max, axis=0))
        self.expected_throughput.assign_add(tf.math.reduce_sum(i_expected, axis=0))
        self.count.assign_add(y_true.shape[0])
    def result(self):
        return self.max_throughput/self.count, self.expected_throughput/self.count
class Regression_ExpectedThroughput(tf.keras.metrics.Metric):
    def __init__(self, name='expected_throughput', **kwargs):
        super(Regression_ExpectedThroughput, self).__init__(name=name, **kwargs)
        self.max_throughput = self.add_weight(name='tp', initializer='zeros')
        self.expected_throughput = self.add_weight(name='tp2', initializer='zeros')
        self.count = self.add_weight(name='tp3', initializer='zeros')
        self.a = tf.math.log(tf.cast(2, dtype=tf.float32))
    def update_state(self, y_true, y_pred, x):
        c_max = tf.gather(x, y_true, axis=1, batch_dims=1)
        i_max = tf.math.log(1 - 1.5*tf.math.log(1 - c_max))/self.a
        y_pred_rounded = tf.cast(tf.math.round(y_pred), tf.int32)
        y_pred_rounded = tf.where(y_pred_rounded >= x.shape[1], x.shape[1]-1, y_pred_rounded)
        y_pred_rounded = tf.where(y_pred_rounded < 0, 0, y_pred_rounded)
        c_pred = tf.gather(x, y_pred_rounded, axis=1, batch_dims=1)
        i_pred = tf.math.log(1 - 1.5 * tf.math.log(1 - c_pred)) / self.a
        self.max_throughput.assign_add(tf.math.reduce_sum(i_max, axis=0))
        self.expected_throughput.assign_add(tf.math.reduce_sum(i_pred, axis=0))
        self.count.assign_add(y_true.shape[0])
    def result(self):
        return self.max_throughput/self.count, self.expected_throughput/self.count
class Regression_Accuracy(tf.keras.metrics.Metric):
    def __init__(self, name='expected_throughput', **kwargs):
        super(Regression_Accuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='tp', initializer='zeros')
        self.count = self.add_weight(name='tp2', initializer='zeros')
    def update_state(self, y_true, y_pred):
        y_rounded_pred = tf.cast(tf.math.round(y_pred), tf.float32)
        diff = y_true - y_rounded_pred
        count = tf.cast(tf.reduce_sum(tf.where(diff == 0, 1, 0)), tf.float32)
        # i_pred = tf.reshape((tf.math.log(1 - 1.5*tf.math.log(1 - c_pred))/self.a), (i_max.shape[0], ))
        self.correct.assign_add(count)
        self.count.assign_add(y_true.shape[0])
    def result(self):
        return self.correct/self.count

class TargetThroughput(tf.keras.metrics.Metric):
    def __init__(self, name='expected_throughput', logit=True, **kwargs):
        super(TargetThroughput, self).__init__(name=name, **kwargs)
        self.max_throughput = self.add_weight(name='tp', initializer='zeros')
        self.expected_throughput = self.add_weight(name='tp2', initializer='zeros')
        self.count = self.add_weight(name='tp3', initializer='zeros')
        self.a = tf.math.log(tf.cast(2, dtype=tf.float32))
        self.logit = logit

    def update_state(self, y_true, y_pred, x):
        c_max = tf.gather(x, y_true, axis=1, batch_dims=1)
        i_max = tf.math.log(1 - 1.5*tf.math.log(1 - c_max))/self.a
        c_picked = tf.argmax(y_pred, axis=1)
        c_picked = tf.gather(x, c_picked, axis=1, batch_dims=1)
        i_expected = tf.math.log(1 - 1.5*tf.math.log(1 - c_picked))/self.a
        self.max_throughput.assign_add(tf.math.reduce_sum(i_max, axis=0))
        self.expected_throughput.assign_add(tf.math.reduce_sum(i_expected, axis=0))
        self.count.assign_add(y_true.shape[0])

    def result(self):
        return self.max_throughput/self.count, self.expected_throughput/self.count
# ============================  Loss fn  ============================
def Encoding_variance():
    def encoding_variance(encode):
        loss = -tf.reduce_sum(tf.math.reduce_variance(encode, axis=0))
        return loss
    return encoding_variance
def Encoding_distance():
    def encoding_distance(encode):
        loss = 0
        for i in range(0, encode.shape[0]-1):
            loss = loss + tf.norm((encode[i], encode[i+1]))
        return -loss/encode.shape[0]
    return encoding_distance
def Loss_LSTM_encoding_diversity():
    model_path = "trained_models/Sept 29/LSTM_Loss_function_no_relu.h5"
    loss_model = tf.keras.models.load_model(model_path)
    for item in loss_model.layers:
        item.trainable = False
    return loss_model
def Regularization_loss():
    def regulariztion_loss(y_pred):
        loss = -tf.reduce_sum(tf.square(y_pred))/(y_pred.shape[0] * y_pred.shape[1])
        return loss
    return regulariztion_loss
def Throughout_diff_Loss():
    def through_put_loss(y_true, y_pred, x=None):
        a = tf.math.log(tf.cast(2, dtype=tf.float32))
        c_max = tf.gather(x, y_true, axis=1, batch_dims=1)
        i_max = tf.math.log(1 - 1.5 * tf.math.log(1 - c_max)) / a
        weighted_c = tf.math.multiply(Softmax()(y_pred), x)
        i_expected = tf.math.reduce_sum(tf.math.log(1 - 1.5 * tf.math.log(1 - weighted_c)) / a, axis=1)
        return tf.square(tf.math.reduce_sum(i_max, axis=0) - tf.math.reduce_sum(i_expected, axis=0))
    return through_put_loss
def ThroughputLoss():
    def through_put_loss(y_true, y_pred, x=None):
        a = tf.math.log(tf.cast(2, dtype=tf.float32))
        max_x = tf.argmax(y_pred, axis=1)
        c_chosen = tf.gather(x, max_x, axis=1, batch_dims=1)
        i_expected = tf.math.log(1 - 1.5 * tf.math.log(1 - c_chosen)) / a,
        cost = -tf.math.reduce_sum(i_expected, axis=0)
        return cost
    return through_put_loss
def ExpectedThroughputLoss():
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

# =========================== Custom function for straight through estimation ============================

@tf.custom_gradient
def sign_relu_STE(x):
    rtv = tf.sign(x)
    def grad(dy):
        one = tf.constant(1, dtype=tf.float32)
        zero = tf.constant(0, dtype=tf.float32)
        # back = tf.maximum(tf.minimum(x, pos), neg)
        back = tf.where(tf.abs(x) >= 0, one, zero)
        grad_val = dy * back
        return grad_val
    return rtv, grad

def hard_tanh(x):
    neg = tf.constant(-1, dtype=tf.float32)
    pos = tf.constant(1, dtype=tf.float32)
    rtv = tf.maximum(tf.minimum(x, pos), neg)
    return rtv
def leaky_hard_tanh(x):
    rtv = tf.maximum(tf.minimum(x, 1.0 + 0.01 * x), -1.0 + 0.01 * x)
    return rtv
def clippedRelu(x):
    return tf.maximum(tf.minimum(0.01 * x, x), 0.01 * x)
def annealing_sigmoid(x, N):
    alpha = tf.minimum(5.0, 1.0 + 0.01*N)
    out = tf.sigmoid(alpha*x)
    return out
def annealing_tanh(x, N):
    alpha = tf.minimum(5.0, 1.0 + 0.01*N)
    out = tf.tanh(alpha*x)
    return out

# ========================================== misc ==========================================
def replace_tanh_with_sign(model, model_func, k):
    model.save_weights('weights.hdf5')
    new_model = model_func((k, ), k, saved=True)
    new_model.load_weights('weights.hdf5')
    return new_model

if __name__ == "__main__":
    gen_encoding_data()