from util import *
from models import *
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
def test_performance(model, M = 20, K = 5, B = 10, N_rf = 5, sigma2_h = 6.3, sigma2_n = 0.00001):
    # tp_fn = ExpectedThroughput(name = "throughput")
    result = np.zeros((3, ))
    loss_fn1 = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    loss_fn2 = Binarization_regularization(K, 1000, M)
    tf.random.set_seed(80)
    print("Testing Starts")
    for e in range(0, 1):
        ds = generate_link_channel_data(1000, K, M)
        # prediction = tf.reshape(model(features), (1,))
        # ds_load = float_to_floatbits(ds, complex=True)
        ds_load = ds

        prediction = model(ds_load, training=False)
        prediction = Masking_with_learned_weights_soft(K, M, sigma2_n)(prediction)
        # print(prediction[1])
        prediction = Harden_scheduling(K, M, sigma2_n, k=N_rf)(prediction)
        result[0] = tf.reduce_mean(loss_fn1(prediction, ds))
        result[1] = loss_fn2(prediction)
        print(result)
        # ========= ========= =========  plotting ========= ========= =========
        ds = tf.square(tf.abs(ds))
        unflattened_X = tf.reshape(prediction, (prediction.shape[0], K, M))
        unflattened_X = tf.transpose(unflattened_X, perm=[0, 2, 1])
        denominator = tf.matmul(ds, unflattened_X)
        for i in range(0, 1000):
            plt.imshow(denominator[i], cmap="gray")
            plt.show(block=False)
            plt.pause(0.0001)
            plt.close()
        # ========= ========= =========  plotting ========= ========= =========
        # submodel = Model(inputs=model.input, outputs=model.get_layer("model (Model)").output)
        # prediction2 = tf.reshape(submodel(ds_load), (1000*M, B))
        # print(prediction2)
def plot_data(arr, col):
    cut = 0
    for i in range(20, arr.shape[0]):
        if arr[i, 0] == 0:
            cut = i
            break
    arr = arr[:i, :]
    x = np.arange(0, arr.shape[0])
    plt.plot(x, arr[:, col])
    plt.title("Regularization Loss")
    plt.show()
if __name__ == "__main__":
    file = "trained_models/Jul 30th/sumrate_VS_softmasking_peruser_softmax_noise=0.1"
    # file = "trained_models/Sept 25/k=2, L=2/Data_gen_encoder_L=1_k=2_tanh_annealing"
    N = 1000
    M = 40
    K = 10
    B = 10
    seed = 200
    check = 100
    N_rf = 3

    sigma2_h = 6.3
    sigma2_n = 0.1
    model_path = file + ".h5"
    training_data_path = file + ".npy"
    # training_data = np.load(training_data_path)
    # plot_data(training_data, 1)
    model = tf.keras.models.load_model(model_path)
    print(model.summary())
    test_performance(model, M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h = sigma2_h)
