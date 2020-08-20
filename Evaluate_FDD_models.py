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
    loss_fn1 = tf.keras.losses.MeanSquaredError()
    # loss_fn1 = Sum_rate_utility_RANKING_hard(K, M, sigma2_n, N_rf, True)
    # loss_fn2 = Binarization_regularization(K, num_data, M, k=N_rf)
    loss_fn2 = Total_activation_limit_hard(K, M, N_rf = 0)
    tf.random.set_seed(80)
    print("Testing Starts")
    k = 5
    for e in range(0, 1):
        ds = generate_link_channel_data(num_data, K, M)
        # ds, angle = generate_link_channel_data_with_angle(1000, K, M)
        ds_load = ds
        # prediction = ensumble_output(ds_load, model, k, loss_fn1) # this outputs (N, M*K, k)
        prediction = model(ds_load)[0]
        print(tf.reduce_sum(prediction[0]))
        out = loss_fn1(prediction, tf.abs(ds_load))
        result[0] = tf.reduce_mean(out)
        result[1] = loss_fn2(prediction)
        print("the soft result is ", result)
        print("the variance is ", tf.math.reduce_std(out))
        A[2]
        prediction_binary = binary_activation(prediction)
        out_binary = loss_fn1(prediction_binary, ds_load)
        result[0] = tf.reduce_mean(out_binary)
        result[1] = loss_fn2(prediction_binary)
        print("the hard result is ", result)
        print("the variance for binary result is ", tf.math.reduce_std(out_binary))

        prediction_hard = Harden_scheduling(k=N_rf)(prediction)
        out_hard = loss_fn1(prediction_hard, ds_load)
        result[0] = tf.reduce_mean(out_hard)
        result[1] = loss_fn2(prediction_hard)
        print("the top Nrf result is ", result)
        print("the variance for hard result is ", tf.math.reduce_std(out_hard))
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
    for i in range(arr.shape[0]-1, 0, -1):
        if arr[i, 0] != 0:
            cut = i
            break
    arr = arr[:i, :]
    x = np.arange(0, arr.shape[0])
    plt.plot(x, arr[:, col])
    # plt.plot(x, arr[:, 3])
    # plt.plot(x, arr[:, 3])
    plt.title("Reconstruction Loss")
    plt.show()
if __name__ == "__main__":
    file = "trained_models/aug20th/B=5,E=30+relu_VAE+qualitybit"
    custome_obj = {'Closest_embedding_layer': Closest_embedding_layer, 'Interference_Input_modification': Interference_Input_modification,
                   'Interference_Input_modification_no_loop': Interference_Input_modification_no_loop,
                   "Interference_Input_modification_per_user":Interference_Input_modification_per_user}
    N = 1000
    M = 40
    K = 10
    B = 3
    seed = 200
    check = 100
    N_rf = 3
    sigma2_h = 6.3
    sigma2_n = 0.1
    tf.random.set_seed(seed)
    np.random.seed(seed)
    model_path = file + ".h5"
    training_data_path = file + ".npy"
    # training_data = np.load(training_data_path)
    # plot_data(training_data, 0)
    # training_data = np.load(training_data_path)
    # plot_data(training_data, 0)
    model = tf.keras.models.load_model(model_path, custom_objects=custome_obj)
    for i in range(3,4):
        N_rf = i
        print("========================================== B =", i)
        # model = partial_feedback_top_N_rf_model(N_rf, B, 1, M, K, sigma2_n)
        print(model.get_layer("encoder_0").summary())
        print(model.get_layer("decoder").summary())
        print(model.summary())
        # model = NN_Clustering(N_rf, M, reduced_dim=8)
        # model = k_clustering_hieristic(N_rf)
        # model = greedy_hieristic(N_rf, sigma2_n)
        # model = top_N_rf_user_model(M, K, N_rf)
        # model = partial_feedback_semi_exhaustive_model(N_rf, B, 2, M, K, sigma2_n)
        # print(model.summary())
        test_performance(model, M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h = sigma2_h)
