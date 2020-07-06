import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Softmax, Input, ThresholdedReLU, Flatten
from tensorflow.keras.activations import sigmoid
from matplotlib import pyplot as plt
from util import *
from models import *
def quantizaton_evaluation(model, granuality = 0.001, k=2):
    submodel = Model(inputs=model.input, outputs=model.get_layer("tf_op_layer_concat").output)
    count = np.arange(0, 1, granuality)
    bin = np.zeros((k, count.shape[0]))
    for i in range(count.shape[0]):
        channel_data = tf.ones((1, k)) * count[i]
        channel_label = tf.math.argmax(channel_data, axis=1)
        channel_data = float_to_floatbits(channel_data)
        ds = Dataset.from_tensor_slices((channel_data, channel_label)).batch(1)
        for features, labels in ds:
            # features_mod = tf.ones((1, 1))
            # features_mod = tf.concat((features_mod, tf.reshape(features, (1, features.shape[1]))), axis=1)
            features_mod = tf.ones((features.shape[0], features.shape[1], 1)) * 1
            features_mod = tf.concat((features_mod, features), axis=2)
            out = submodel(features_mod)
            bin = log_bin(bin, out, i, k)
    bin[1, :] = (bin[1, :] - 2)
    plt_2D_quantization_comparison(bin, granuality)
    # plt.plot(count, bin[0, :])
    # plt.plot(count, bin[1, :])
    # plt.show()
    # layer_output = model.get_layer(layer_name).output
def plt_2D_quantization_comparison(bin, granuality=0.001):
    arr = np.zeros((bin.shape[1], bin.shape[1], 3))
    for i in range(bin.shape[1]):
        for j in range(bin.shape[1]):
            if bin[0, i] > bin[1, j]:
                arr[i, j, 0] = 1
            elif bin[0, i] < bin[1, j]:
                arr[i, j, 0] = 1
            elif bin[0, i] <=0 and bin[1, j] <=0:
                arr[i, j, 1] = 1
                # arr[i, j, 0] = 1
            elif bin[0, i] > 0 and bin[1, j] > 0:
                arr[i, j, 2] = 1
                # arr[i, j, 0] = 1
    print(arr[:,:,0].sum()/arr.shape[0]/arr.shape[0]/2)
    arr = np.flip(arr, axis=0)
    plt.imshow(arr)
    plt.show()

def comparison_evaluation(model, l, k=2):
    submodel = Model(inputs=model.get_layer("dense_1").input, outputs=model.output)
    input = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]])
def log_bin(bin, output, n, k):
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
def quantization_evaluation_regression(model, granuality = 0.0001):
    count = np.arange(0, 1, 0.001)
    out = np.zeros(count.shape)
    j = 0
    for i in count:
        data_set = tf.ones((1,)) * i
        input = float_to_floatbits(data_set)
        input = tf.concat((tf.reshape(data_set, (1,1)), input), axis=1)
        tim = model(input)
        out[j] = tim[0]
        j += 1
    plt.plot(count, out)
    plt.xlabel("input")
    plt.ylabel("output")
    plt.show()
def variance_graph(model, N = 10000):
    # tp_fn = ExpectedThroughput(name = "throughput")
    tp_fn = TargetThroughput(name = "target throughput")
    # tp_fn = ExpectedThroughput(name = "target throughput")
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
        ds = gen_channel_quality_data_float_encoded(1, k=2)
        for features, labels in ds:
            # prediction = tf.reshape(model(features), (1,))
            features_mod = tf.ones((features.shape[0], features.shape[1], 1)) * 1
            features_mod = tf.concat((features_mod, features), axis=2)
            prediction = model(features_mod)
            # tp_fn(labels, prediction, features)
            a_fn(labels, prediction)
            # if a_fn.result() != 1:
            #     print(features, '\n', prediction)
            # if e%1000 == 0:
            #     template = "At Test case {}, The The maximum Throughput is {}, Expected Throughout is {}, the Accuracy is {}"
            #     print(template.format(e, tp_fn.result()[0], tp_fn.result()[1], a_fn.result()))
        acc[e] = a_fn.result()
        result[e, 0] = tp_fn.result()[0]
        result[e, 1] = tp_fn.result()[1]
    print("Max Throughput:", np.nanmean(result[:, 0]))
    print("Max Throughput Variance:", np.nanvar(result[:, 0]))
    print("Expected Throughput:", np.nanmean(result[:, 1]))
    print("Expected Throughput Variance:", np.nanvar(result[:, 1]))
    print("Accuracy:", np.nanmean(acc))
    print("Accuracy Variance:", np.nanvar(acc))
def variance_graph_accuracy(model, N = 10000):
    # tp_fn = ExpectedThroughput(name = "throughput")
    # tp_fn = ExpectedThroughput(name = "target throughput")
    # a_fn = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    a_fn = tf.keras.metrics.MeanSquaredError()
    # tp_fn = Regression_ExpectedThroughput()
    # a_fn = Regression_Accuracy()
    result = np.zeros((N,2))
    acc = np.zeros((N, ))
    tf.random.set_seed(80)
    print("Testing Starts")
    for e in range(0, N):
        a_fn.reset_states()
        # ds = gen_data(1, k=10, batchsize=1)
        ds = gen_regression_data(N=1, batchsize=1, reduncancy=9)
        for features, labels in ds:
            # prediction = tf.reshape(model(features), (1,))
            features_mod = tf.ones((features.shape[0], 1)) * e
            features_mod = tf.concat((features_mod, features), axis=1)
            prediction = model(features_mod)
            a_fn(labels, prediction)
            # if a_fn.result() != 1:
            #     print(features, '\n', prediction)
            # if e%1000 == 0:
            #     template = "At Test case {}, The The maximum Throughput is {}, Expected Throughout is {}, the Accuracy is {}"
            #     print(template.format(e, tp_fn.result()[0], tp_fn.result()[1], a_fn.result()))
        acc[e] = a_fn.result()
    print("Accuracy:", np.nanmean(acc))
    print("Accuracy Variance:", np.nanvar(acc))
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
    cut = 0
    for i in range(20, arr.shape[0]):
        if arr[i, 0] == 0:
            cut = i
            break
    arr = arr[:i, :]
    x = np.arange(0, arr.shape[0])
    error = arr[0]
    # plt.plot(x, arr[:,3])
    # plt.plot(x, arr[:,7])
    # plt.plot(x, arr[:,2])
    plt.plot(x, arr[:, 0])
    # plt.plot(x, arr[:, 3])
    plt.title("Loss")
    # plt.legend(("Training", "Test", "Maximum"))
    # plt.legend(("loss"))
    # plt.legend(("Quantization Level Count"))
    plt.show()
if __name__ == "__main__":
    file = "trained_models/Jul 3nd/2_user_1_qbit_small_4_layer_deep_encoder_tanh(relu)"
    # file = "trained_models/Sept 25/k=2, L=2/Data_gen_encoder_L=1_k=2_tanh_annealing"
    model_path = file + ".h5"
    training_data_path = file + ".npy"
    # training_data_path1 = file + "_cont.npy"
    # training_data_path2 = file + "_cont2.npy"
    # training_data = np.concatenate((np.load(training_data_path), np.load(training_data_path1)), axis=0)
    training_data = np.load(training_data_path)
    model = tf.keras.models.load_model(model_path)
    print(model.summary())
    # model = create_uniformed_quantization_model(k=2, bin_num=2)
    # plot_data(training_data)
    quantizaton_evaluation(model)
    variance_graph(model, N=10000)
    # variance_graph_accuracy(model, N=1000)
    # quantization_evaluation_regression(model)

