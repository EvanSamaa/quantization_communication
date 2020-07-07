import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Softmax, Input, ThresholdedReLU, Flatten
from tensorflow.keras.activations import sigmoid
import random
import math
# ============================  Data gen ============================

def gen_data(N, k, low=0, high=1, batchsize=30):
    channel_data = tf.random.uniform((N,k), low, high)
    channel_label = tf.math.argmax(channel_data, axis=1)
    dataset = Dataset.from_tensor_slices((channel_data, channel_label)).batch(batchsize)
    return dataset
def gen_channel_quality_data_float_encoded(N, k, low=0, high=1):
    channel_data = tf.random.uniform((N, k), low, high)
    channel_label = tf.math.argmax(channel_data, axis=1)
    channel_data = float_to_floatbits(channel_data)
    dataset = Dataset.from_tensor_slices((channel_data, channel_label)).batch(N)
    return dataset
def gen_number_data(N=10000, k = 7.5, batchsize=10000):
    channel_data_num = tf.random.uniform((N, 1), 0, k)
    channel_data_num = tf.cast(tf.round(channel_data_num), dtype=tf.int32)
    # channel_data_num = tf.round(channel_data_num)
    channel_data = tf.cast(tf.one_hot(channel_data_num, depth=math.ceil(k), on_value=1.0, off_value=0.0), tf.float32)
    channel_data = tf.reshape(channel_data, (N, math.ceil(k)))
    channel_label = channel_data_num
    dataset = Dataset.from_tensor_slices((channel_data
                                          , channel_label)).batch(batchsize)
    return dataset
def gen_encoding_data(N=1000, Sequence_length=10000, k=16, batchsize = 100, bit = 4):
    # dict_list = [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]
    dict_list = [[-1, -1, -1, -1], [-1, -1, -1, 1], [-1, -1, 1, -1], [-1, -1, 1, 1],
                 [-1, 1, -1, -1], [-1, 1, -1, 1], [-1, 1, 1, -1], [-1, 1, 1, 1],
                 [1, -1, -1, -1], [1, -1, -1, 1], [1, -1, 1, -1], [1, -1, 1, 1],
                 [1, 1, -1, -1], [1, 1, -1, 1], [1, 1, 1, -1], [1, 1, 1, 1]]
    output = np.zeros((N, Sequence_length, bit))
    channel_label = np.zeros((N, 1))
    for n in range(N):
        random.shuffle(dict_list)
        current = random.randint(1,k)
        current_encoding = np.array(dict_list[:current])
        idx = np.random.randint(0, current, size=Sequence_length)
        for m in range(Sequence_length):
            output[n, m] = current_encoding[idx[m]]
        channel_label[n] = current
    channel_label = tf.cast(channel_label, tf.float32)
    output = tf.cast(output, tf.float32)
    dataset = Dataset.from_tensor_slices((output, channel_label)).batch(batchsize)
    return dataset
def gen_regression_data(N=10000, batchsize=10000, reduncancy=1):
    ################
    data_set = tf.random.uniform((N, ), 0, 1)
    label_set = data_set
    modified_dataset = float_to_floatbits(data_set)
    ################
    # ones = tf.ones((N, reduncancy))
    # data_set = tf.random.uniform((N,1), 0, 1) # for redundancy
    # label_set = data_set
    # data_set = tf.concat((data_set, -data_set, tf.exp(data_set), tf.square(data_set)), axis=1)
    # data_set = tf.multiply(ones, data_set)
    # print(data_set)
    dataset = Dataset.from_tensor_slices((modified_dataset, label_set)).batch(batchsize)
    return dataset
# ============================  Metrics  ============================
def quantizaton_evaluation_numbers(model, granuality=0.0001, k=2):
    tsub_model = Model(inputs=model.input, outputs=model.get_layer("tf_op_layer_Sign").output)
    for i in range(0, 16):
        features = tf.ones((1,1))*i
        # features = tf.one_hot([0, 1, 2, 3, 4, 5, 6, 7], depth=8)[i]
        features_mod = tf.ones((1, 1))
        features_mod = tf.concat((features_mod, tf.reshape(features, (1, features.shape[0]))), axis=1)
        out = tsub_model(features_mod)
        out_2 = model(features_mod)
        # print(i, out, Softmax()(out_2))
        print(i, out)
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
def Quantization_count(x):
    # pass in (N, encoding_size) array to delete duplicate on the first axis
    x_shape = tf.shape(x)  # (3,2)
    x1 = tf.tile(x, [1, x_shape[0]])  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]
    x2 = tf.tile(x, [x_shape[0], 1])  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]

    x1_2 = tf.reshape(x1, [x_shape[0] * x_shape[0], x_shape[1]])
    x2_2 = tf.reshape(x2, [x_shape[0] * x_shape[0], x_shape[1]])
    cond = tf.reduce_all(tf.equal(x1_2, x2_2), axis=1)
    cond = tf.reshape(cond, [x_shape[0], x_shape[0]])  # reshaping cond to match x1_2 & x2_2
    cond_shape = tf.shape(cond)
    cond_cast = tf.cast(cond, tf.int32)  # convertin condition boolean to int
    cond_zeros = tf.zeros(cond_shape, tf.int32)  # replicating condition tensor into all 0's

    # CREATING RANGE TENSOR
    r = tf.range(x_shape[0])
    r = tf.add(tf.tile(r, [x_shape[0]]), 1)
    r = tf.reshape(r, [x_shape[0], x_shape[0]])

    # converting TRUE=1 FALSE=MAX(index)+1 (which is invalid by default) so when we take min it wont get selected & in end we will only take values <max(indx).
    f1 = tf.multiply(tf.ones(cond_shape, tf.int32), x_shape[0] + 1)
    f2 = tf.ones(cond_shape, tf.int32)
    cond_cast2 = tf.where(tf.equal(cond_cast, cond_zeros), f1, f2)  # if false make it max_index+1 else keep it 1

    # multiply range with new int boolean mask
    r_cond_mul = tf.multiply(r, cond_cast2)
    r_cond_mul2 = tf.reduce_min(r_cond_mul, axis=1)
    r_cond_mul3, unique_idx = tf.unique(r_cond_mul2)
    r_cond_mul4 = tf.subtract(r_cond_mul3, 1)

    # get actual values from unique indexes
    op = tf.gather(x, r_cond_mul4)

    return (op.shape[0])
class TargetThroughput(tf.keras.metrics.Metric):
    def __init__(self, name='expected_throughput', logit=True, bit_string = True, **kwargs):
        super(TargetThroughput, self).__init__(name=name, **kwargs)
        self.max_throughput = self.add_weight(name='tp', initializer='zeros')
        self.expected_throughput = self.add_weight(name='tp2', initializer='zeros')
        self.count = self.add_weight(name='tp3', initializer='zeros')
        self.a = tf.math.log(tf.cast(2, dtype=tf.float32))
        self.logit = logit
        self.bit_string = bit_string

    def update_state(self, y_true, y_pred, x):
        if self.bit_string:
            mod_x = floatbits_to_float(x)
        else:
            mod_x = x
        c_max = tf.gather(x, y_true, axis=1, batch_dims=1)
        i_max = tf.math.log(1 - 1.5*tf.math.log(1 - c_max))/self.a
        c_picked = tf.argmax(y_pred, axis=1)
        c_picked = tf.gather(mod_x, c_picked, axis=1, batch_dims=1)
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
def Loss_NN_encoding_diversity():
    model_path = "trained_models/Encoding Diversity/MLP_loss_function_2.h5"
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
def Negative_shove():
    def negative_shove(y_pred, x=None):
        values, indices = tf.nn.top_k(y_pred, 2)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(values[:,1], y_pred)
        return -loss
    return negative_shove


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
def binary_activation(x):
    out = tf.maximum(tf.sign(x), 0)
    return out
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
def annealing_tanh(x, N, name):
    alpha = tf.minimum(5.0, 1.0 + 0.01*N)
    out = tf.tanh(alpha*x, name=name)
    return out

# ========================================== misc ==========================================
def replace_tanh_with_sign(model, model_func, k):
    model.save_weights('weights.hdf5')
    new_model = model_func((k, ), k, saved=True)
    new_model.load_weights('weights.hdf5')
    return new_model
# def float_bits_to_float(bits_arr):
#     cp_value_arr = bits_arr.numpy()
#     out = np.zeros((bits_arr.shape[0], 1))
#     for j in range(0, bits_arr.shape[1]):
#         for i in range(0, 23):
#             cp_value_arr[:, j] = cp_value_arr[:, j] * 2
#             out[:, j, i] = np.where(cp_value_arr[:, j] >= 1, 1, 0)
#             cp_value_arr[:, j] = cp_value_arr[:, j] - out[:, j, i]
#     return tf.constant(out, dtype=tf.float32)
def floatbits_to_float(value_arr):
    np_value_arr = value_arr.numpy()
    if len(value_arr.shape) == 3:
        out = np.zeros((value_arr.shape[0], value_arr.shape[1]))
        for i in range(0, 23):
            out = out + np_value_arr[:, :, i] * np.float_power(2, -(i+1))
        return tf.constant(out, dtype=tf.float32)
def float_to_floatbits(value_arr):
    cp_value_arr = value_arr.numpy()
    # I'm not sure if the shape thing works well so might have to comeback and fix it
    if len(value_arr.shape) == 1:
        out = np.zeros((value_arr.shape[0], 23))
        for i in range(0, 23):
            cp_value_arr = cp_value_arr*2
            out[:, i] = np.where(cp_value_arr >= 1, 1, 0)
            cp_value_arr = cp_value_arr - out[:, i]
        return tf.constant(out, dtype=tf.float32)
    else:
        out = np.zeros((value_arr.shape[0], value_arr.shape[1], 23))
        for j in range(0, value_arr.shape[1]):
            for i in range(0, 23):
                cp_value_arr[:, j] = cp_value_arr[:, j] * 2
                out[:, j, i] = np.where(cp_value_arr[:, j] >= 1, 1, 0)
                cp_value_arr[:, j] = cp_value_arr[:, j]- out[:, j, i]
        return tf.constant(out, dtype=tf.float32)
if __name__ == "__main__":
