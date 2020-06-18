import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Softmax, Input, ThresholdedReLU, Flatten
from tensorflow.keras.activations import sigmoid
from matplotlib import pyplot as plt
class ExpectedThroughput(tf.keras.metrics.Metric):
    def __init__(self, name='expected_throughput', **kwargs):
        super(ExpectedThroughput, self).__init__(name=name, **kwargs)
        self.max_throughput = self.add_weight(name='tp', initializer='zeros')
        self.expected_throughput = self.add_weight(name='tp2', initializer='zeros')
        self.a = tf.math.log(tf.cast(2, dtype=tf.float32))
    def update_state(self, y_true, y_pred, x):
        c_max = tf.gather(x, y_true, axis=1, batch_dims=1)
        i_max = tf.math.log(1 - 1.5*tf.math.log(1 - c_max))/self.a
        weighted_c = tf.math.multiply(Softmax()(y_pred), x)
        i_expected = tf.math.reduce_sum(tf.math.log(1 - 1.5*tf.math.log(1 - weighted_c))/self.a, axis=1)
        self.max_throughput.assign_add(tf.math.reduce_sum(i_max, axis=0))
        self.expected_throughput.assign_add(tf.math.reduce_sum(i_expected, axis=0))
    def result(self):
        return self.max_throughput, self.expected_throughput
def gen_data(N, k, low=0, high=1, batchsize=30):
    channel_data = tf.random.uniform((N,k), low, high)
    channel_label = tf.math.argmax(channel_data, axis=1)
    dataset = Dataset.from_tensor_slices((channel_data, channel_label)).batch(batchsize)
    return dataset
def ranking_transform(x):
    out = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for k in range(x.shape[0]):
        for i in range(0, x.shape[1]):
            for j in range(0, x.shape[1]):
                if x[k, i] >= x[k, j]:
                    out[k, i, j] = 1
    return tf.convert_to_tensor(out, dtype=tf.float32)

def variance_graph(model):

    tp_fn = ExpectedThroughput(name = "throughput")
    a_fn = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    result = np.zeros((10000,2))
    acc = np.zeros((10000, ))
    for e in range(0, 10000):
        tp_fn.reset_states()
        a_fn.reset_states()
        ds = gen_data(1, k=10, batchsize=1)
        for features, labels in ds:
            prediction = Softmax()(model(features))
            tp_fn(labels, prediction, features)
            a_fn(labels, prediction)
            if a_fn.result() != 1:
                print(features, '\n', prediction)
        result[e, 0] = tp_fn.result()[0]
        result[e, 1] = tp_fn.result()[1]
        acc[e] = a_fn.result()
    tp_diff = (result[:, 0] - result[:, 1])
    print(result[:, 0].mean())
    print(result[:, 0].var())
    print(tp_diff.mean())
    print(tp_diff.var())
    print(acc.mean())
    print(acc.var())



if __name__ == "__main__":
    model_throughput = tf.keras.models.load_model("models/three_layer_MLP_throughput_loss.h5")
    # model_throughput = tf.keras.models.load_model("models/three_layer_MLP_1800_epoch.h5")
    # model_rounding = tf.keras.models.load_model("models/three_layer_with_rounding_500E.h5")
    # model_compare = tf.keras.models.load_model("models/three_layer_with_pro_processing_10E.h5")
    variance_graph(model_throughput)

