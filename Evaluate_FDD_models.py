from util import *
from models import *
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
def test_performance(model, M = 20, K = 5, B = 10, N_rf = 5, sigma2_h = 6.3, sigma2_n = 0.00001):
    # tp_fn = ExpectedThroughput(name = "throughput")
    num_data = 1000
    result = np.zeros((3, ))
    loss_fn1 = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    # loss_fn1 = Sum_rate_utility_RANKING_hard(K, M, sigma2_n, N_rf, True)
    loss_fn2 = Binarization_regularization(K, num_data, M, k=N_rf)
    loss_fn2 = Total_activation_limit_soft(K, M, N_rf = 0)
    tf.random.set_seed(80)
    print("Testing Starts")
    for e in range(0, 1):
        ds = generate_link_channel_data(num_data, K, M)
        # ds, angle = generate_link_channel_data_with_angle(1000, K, M)
        # angle = tf.reshape(angle, (1000, K))
        # prediction = tf.reshape(model(features), (1,))
        # ds_load = float_to_floatbits(ds, complex=True)
        ds_load = ds
        prediction = model(ds_load)
        prediction = Harden_scheduling(k=N_rf)(prediction)
        result[0] = tf.reduce_mean(loss_fn1(prediction, ds))
        result[1] = tf.sqrt(tf.reduce_mean(loss_fn2(prediction)))
        print(result)
        # ========= ========= =========  plotting ========= ========= =========
        # ds = tf.square(tf.abs(ds))
        # # prediction = prediction[:, :, 2]
        # unflattened_X = tf.reshape(prediction, (prediction.shape[0], K, M))
        # unflattened_X = tf.transpose(unflattened_X, perm=[0, 2, 1])
        # denominator = tf.matmul(ds, unflattened_X)
        # for i in range(0, num_data):
        #     plt.imshow(denominator[i], cmap="gray")
        #     plt.show(block=False)
        #     plt.pause(0.0001)
        #     plt.close()
        # ========= ========= =========  plotting ========= ========= =========
def plot_data(arr, col):
    cut = 0
    for i in range(20, arr.shape[0]):
        if arr[i, 0] == 0:
            cut = i
            break
    arr = arr[:i, :]
    x = np.arange(0, arr.shape[0])
    plt.plot(x, arr[:, col])
    plt.title("Penalty")
    plt.show()
if __name__ == "__main__":
    file = "trained_models/Aug9th/Wei_cui_like_model_with_softmax_k=8"
    custome_obj = {'Closest_embedding_layer': Closest_embedding_layer, 'Interference_Input_modification': Interference_Input_modification}
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
    # plot_data(training_data, 2)
    # training_data = np.load(training_data_path)
    # plot_data(training_data, 0)
    model = tf.keras.models.load_model(model_path, custom_objects=custome_obj)
    print(model.summary())
    # model = NN_Clustering(N_rf, M, reduced_dim=8)
    # model = k_clustering_hieristic(N_rf)
    # model = greedy_hieristic(N_rf, sigma2_n)
    # model = top_N_rf_user_model(M, K, N_rf)
    # print(model.summary())
    test_performance(model, M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h = sigma2_h)
