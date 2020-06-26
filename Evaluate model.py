import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Softmax, Input, ThresholdedReLU, Flatten
from tensorflow.keras.activations import sigmoid
from matplotlib import pyplot as plt
from util import *
from models import *
def quantizaton_evaluation(model, granuality = 0.0001, k=2):
    submodel = Model(inputs=model.input, outputs=model.get_layer("tf_op_layer_concat").output)
    count = np.arange(0, 1, 0.0001)
    bin = np.zeros((k, count.shape[0]))
    for i in range(count.shape[0]):
        channel_data = tf.ones((1, k)) * count[i]
        channel_label = tf.math.argmax(channel_data, axis=1)
        ds = Dataset.from_tensor_slices((channel_data, channel_label))
        for features, labels in ds:
            features_mod = tf.ones((1, 1))
            features_mod = tf.concat((features_mod, tf.reshape(features, (1, features.shape[0]))), axis=1)
            out = submodel(features_mod)
            bin = log_bin(bin, out, i)
    plt.plot(count, bin[0, :])
    plt.plot(count, bin[1, :])
    plt.show()
    # layer_output = model.get_layer(layer_name).output
def log_bin(bin, output, n):
    for i in range(output.shape[1]):
        bin[i, n] = get_num_from_binary(output[:, i]) + i*2**output.shape[0]
    return bin
def get_num_from_binary(binary):
    power = 0
    sum = 0
    for i in range(binary.shape[0]-1, -1, -1):
        if binary[i] > 0:
            sum = sum + 2**power
        power = power + 1
    return sum


def variance_graph(model, N = 10000):
    # tp_fn = ExpectedThroughput(name = "throughput")
    tp_fn = TargetThroughput(name = "target throughput")
    a_fn = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    # tp_fn = Regression_ExpectedThroughput()
    # a_fn = Regression_Accuracy()
    result = np.zeros((N,2))
    acc = np.zeros((N, ))
    tf.random.set_seed(80)
    print("Testing Starts")
    for e in range(0, N):
        tp_fn.reset_states()
        a_fn.reset_states()
        ds = gen_data(1, k=10, batchsize=1)
        for features, labels in ds:
            # prediction = tf.reshape(model(features), (1,))
            prediction = model(features)
            tp_fn(labels, prediction, features)
            a_fn(labels, prediction)
            if a_fn.result() != 1:
                print(features, '\n', prediction)
            # if e%1000 == 0:
            #     template = "At Test case {}, The The maximum Throughput is {}, Expected Throughout is {}, the Accuracy is {}"
            #     print(template.format(e, tp_fn.result()[0], tp_fn.result()[1], a_fn.result()))
        result[e, 0] = tp_fn.result()[0]
        result[e, 1] = tp_fn.result()[1]
        acc[e] = a_fn.result()
    print("Max Throughput:", result[:, 0].mean())
    print("Max Throughput Variance:", result[:, 0].var())
    print("Expected Throughput:", result[:, 1].mean())
    print("Expected Throughput Variance:", result[:, 1].var())
    print("Accuracy:", acc.mean())
    print("Accuracy Variance:", acc.var())

def three_run_avg(model, N=3000):
    tf.random.set_seed(80)
    tp_fn = ExpectedThroughput(name="throughput")
    a_fn = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    result = np.zeros((N, 2))
    acc = np.zeros((N,))
    tf.random.set_seed(80)
    print("Testing Starts")
    for e in range(0, N):
        tp_fn.reset_states()
        a_fn.reset_states()
        ds = gen_data(1, k=10, batchsize=1)
        for features, labels in ds:
            features_r = tf.reverse(features, axis=[1])
            features_split = tf.concat((features[:, 5:, ], features[:, 0: 5]), axis=1)
            prediction = Softmax()(model(features))
            prediction_r = tf.reverse(Softmax()(model(features_r)), axis=[1])
            prediction_split = Softmax()(model(features_split))

            # prediction = Softmax()(model(ranking_transform(features)))
            # prediction_r = tf.reverse(Softmax()(model(ranking_transform(features_r))), axis=[1])
            # prediction_split = Softmax()(model(ranking_transform(features_split)))

            prediction_split = tf.concat((prediction_split[:, 5:, ], prediction_split[:, 0: 5]), axis=1)
            prediction = (prediction + prediction_r + prediction_split)/3
            tp_fn(labels, prediction, features)
            a_fn(labels, prediction)
            # if a_fn.result() != 1:
            #     print(features, '\n', prediction)
            if e % 1000 == 0:
                template = "At Test case {}, The The maximum Throughput is {}, Expected Throughout is {}, the Accuracy is {}"
                print(template.format(e, tp_fn.result()[0], tp_fn.result()[1], a_fn.result()))
        result[e, 0] = tp_fn.result()[0]
        result[e, 1] = tp_fn.result()[1]
        acc[e] = a_fn.result()
    print("Max Throughput:", result[:, 0].mean())
    print("Max Throughput Variance:", result[:, 0].var())
    print("Expected Throughput:", result[:, 1].mean())
    print("Expected Throughput Variance:", result[:, 1].var())
    print("Accuracy:", acc.mean())
    print("Accuracy Variance:", acc.var())

def plot_data(arr):
    x = np.arange(0, arr.shape[0])
    error = arr[0]
    # plt.plot(x, arr[:,3])
    # plt.plot(x, arr[:,7])
    # plt.plot(x, arr[:,2])
    plt.plot(x, arr[:, 1])
    plt.plot(x, arr[:, 5])
    plt.title("Accuracy")
    # plt.legend(("Training", "Test", "Maximum"))
    plt.legend(("Training", "Test"))
    plt.show()
if __name__ == "__main__":
    file = "trained_models/Sept 25/k=2, L=2/Data_gen_encoder_L=1_k=2_tanh_annealing_2"
    model_path = file + ".h5"
    # training_data_path = file + ".npy"
    # training_data_path1 = file + "_cont.npy"
    # training_data_path2 = file + "_cont2.npy"
    # training_data = np.concatenate((np.load(training_data_path), np.load(training_data_path1)), axis=0)
    model = tf.keras.models.load_model(model_path)
    # model = create_uniformed_quantization_model(k=10, bin_num=2*10)
    # plot_data(training_data)
    # print(model.summary())
    # variance_graph(model, N=1000)
    quantizaton_evaluation(model)

