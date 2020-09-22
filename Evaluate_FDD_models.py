from util import *
from models import *
import numpy as np
# from scipy.io import savemat
import tensorflow as tf
# from matplotlib import pyplot as plt
def test_greedy(model, M = 20, K = 5, B = 10, N_rf = 5, sigma2_h = 6.3, sigma2_n = 0.00001):
    num_data = 1000
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # tp_fn = ExpectedThroughput(name = "throughput")
    result = np.zeros((3,))
    loss_fn1 = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    loss_fn2 = Total_activation_limit_hard(K, M, N_rf=0)
    print("Testing Starts")
    ds_load = generate_link_channel_data(num_data, K, M, 1)
    prediction = model(ds_load)
    counter = 1
    for i in prediction:
        i_complex = tf.complex(tf.sqrt(counter*1.0), 0.0)
        out = loss_fn1(i, tf.abs(ds_load/i_complex))
        result[0] = tf.reduce_mean(out)
        result[1] = loss_fn2(i)
        print("the soft result is ", result)
        print("the variance is ", tf.math.reduce_std(out))
        counter = counter + 1
def test_greedy_different_resolution(M = 20, K = 5, B = 10, N_rf = 5, sigma2_h = 6.3, sigma2_n = 0.00001):
    num_data = 1000
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    # tp_fn = ExpectedThroughput(name = "throughput")
    result = np.zeros((64,))
    print("Testing Starts")
    ds = generate_link_channel_data(num_data, K, M, Nrf=N_rf)
    for i in range(1, 64+1):
        loss_fn1 = Sum_rate_utility_WeiCui(K, M, sigma2_n)
        model = partial_feedback_pure_greedy_model_not_perfect_CSI_available(N_rf, 32, i, M, K, sigma2_n)
        output = model(ds)
        out = loss_fn1(output, tf.abs(ds))
        result[i-1] = tf.reduce_mean(out)
        print("the result for {} is ".format(i), out)
        np.save("trained_models/Sept14th/greedy_resolution_change.npy", result)
def test_DNN_different_K(file_name, M = 20, K = 5, B = 10, N_rf = 5, sigma2_h = 6.3, sigma2_n = 0.00001):
    num_data = 10
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # tp_fn = ExpectedThroughput(name = "throughput")
    result = np.zeros((3,))
    print("Testing Starts")
    ds = generate_link_channel_data(num_data, 60, M)
    Ks = [10, 20, 30, 40, 50, 60]
    for i in Ks:
        ds_load = ds[:, :i]
        loss_fn1 = Sum_rate_utility_WeiCui(i, M, sigma2_n)
        loss_fn2 = Total_activation_limit_hard(i, M, N_rf=0)
        model = tf.keras.models.load_model(file_name.format(i), custom_objects=custome_obj)
        prediction = model.predict(ds_load)[0][:, -1]
        out = loss_fn1(prediction, tf.abs(ds_load))
        result[0] = tf.reduce_mean(out)
        result[1] = loss_fn2(prediction)
        print("the soft result is ", result)
        print("the variance is ", tf.math.reduce_std(out))

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
def different_greedy(m1, m2, M = 20, K = 5, B = 10, N_rf = 5, sigma2_h = 6.3, sigma2_n = 0.00001):
    num_data = 100
    loss_fn1 = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    ds = generate_link_channel_data(num_data, K, M)
    max_difference = 0
    worstG = None
    for i in range(num_data):
        gain_1 = loss_fn1(m1(ds[i:i+1]), ds[i:i+1])
        gain_2 = loss_fn1(m2(ds[i:i+1]), ds[i:i+1])
        difference = tf.abs(tf.abs(gain_1) - tf.abs(gain_2))
        if difference >= max_difference:
            max_difference = difference
            worstG = ds[i]
        if difference > 11:
            break
    print(worstG)
    np.save("G_with_most_difference.npy", np.array(worstG))
    # savemat("G_with_most_difference.mat", {"G":np.array(worstG)})
def test_performance(model, M = 20, K = 5, B = 10, N_rf = 5, sigma2_h = 6.3, sigma2_n = 0.00001):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # tp_fn = ExpectedThroughput(name = "throughput")

    num_data = 1000
    result = np.zeros((3, ))
    loss_fn1 = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    # loss_fn1 = tf.keras.losses.MeanSquaredError()
    # loss_fn1 = Sum_rate_utility_RANKING_hard(K, M, sigma2_n, N_rf, True)
    # loss_fn2 = Bin arization_regularization(K, num_data, M, k=N_rf)
    loss_fn2 = Total_activation_limit_hard(K, M, N_rf = 0)
    print("Testing Starts")
    for e in range(0, 1):
        ds = generate_link_channel_data(num_data, K, M, N_rf)
        # ds, angle = generate_link_channel_data_with_angle(num_data, K, M)
        # print(ds)
        ds_load = ds
        # prediction = ensumble_output(ds_load, model, k, loss_fn1) # this outputs (N, M*K, k)
        prediction = model.predict(ds_load, batch_size=10)[0][:, -1]
        # prediction = model(ds_load)
        # for k in range(0, 10):
        #     plt.plot(np.arange(0, K*M), prediction[k])
        #     plt.show()
        # prediction = model(ds_load)
        out = loss_fn1(prediction, tf.abs(ds_load))
        result[0] = tf.reduce_mean(out)
        result[1] = loss_fn2(prediction)
        print("the soft result is ", result)
        print("the variance is ", tf.math.reduce_std(out))

        prediction_binary = binary_activation(prediction)
        out_binary = loss_fn1(prediction_binary, ds_load)
        result[0] = tf.reduce_mean(out_binary)
        result[1] = loss_fn2(prediction_binary)
        print("the hard result is ", result)
        print("the variance for binary result is ", tf.math.reduce_std(out_binary))

        prediction_hard = Harden_scheduling_user_constrained(N_rf, K, M)(prediction)
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
def plot_data(arr, col=[], title="loss"):
    cut = 0
    for i in range(arr.shape[0]-1, 0, -1):
        if arr[i, 0] != 0:
            cut = i
            break
    arr = arr[:i, :]
    x = np.arange(0, arr.shape[0])
    for i in col:
        plt.plot(x, arr[:, i])
    # plt.plot(x, arr[:, 3])
    plt.title(title)
    plt.show()
if __name__ == "__main__":

    file = "trained_models/Sept14th/perfect_CSI_fixed_concat_X_compress_XG_alt/Perfect_CSI Nrf=8, 2x512_per_linkx6_alt+weighted_CE_loss.h5"
    custome_obj = {'Closest_embedding_layer': Closest_embedding_layer, 'Interference_Input_modification': Interference_Input_modification,
                   'Interference_Input_modification_no_loop': Interference_Input_modification_no_loop,
                   "Interference_Input_modification_per_user":Interference_Input_modification_per_user,
                   "Closest_embedding_layer_moving_avg":Closest_embedding_layer_moving_avg,
                   "Per_link_Input_modification_more_G":Per_link_Input_modification_more_G,
                   "Per_link_Input_modification_more_G_less_X":Per_link_Input_modification_more_G_less_X,
                   "Per_link_Input_modification_even_more_G":Per_link_Input_modification_even_more_G,
                   "Per_link_Input_modification_compress_XG":Per_link_Input_modification_compress_XG,
                   "Per_link_Input_modification_compress_XG_alt": Per_link_Input_modification_compress_XG_alt}
    N = 1
    M = 64
    K = 50
    B = 32
    seed = 200
    check = 100
    N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1
    tf.random.set_seed(seed)
    np.random.seed(seed)
    model_path = file + ".h5"
    # training_data_path = file + ".npy"
    # training_data = np.load(training_data_path)
    # plot_data(training_data, [0, 3], "-sum rate")
    # training_data = np.load(training_datsa_path)
    # plot_data(training_data, 0)
    # model = tf.keras.models.load_model(model_path, custom_objects=custome_obj)
    # N_rfs = [2, 3, 4, 5, 6]
    # model = DP_partial_feedback_semi_exhaustive_model(N_rf, 32, 10, M, K, sigma2_n)
    # test_greedy(model, M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h = sigma2_h)
    mores = [2,3]
    Es = [0]
    # model = DP_partial_feedback_pure_greedy_model(N_rf, B, 10, M, K, sigma2_n, perfect_CSI=True)
    # test_greedy(model, M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h=sigma2_h)
    for j in Es:
        for i in mores:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            print("========================================== E =", j, "more = ", i)
            model = tf.keras.models.load_model(model_path, custom_objects=custome_obj)
            # model = partial_feedback_top_N_rf_model(N_rf, B, 1, M, K, sigma2_n)
            #     print(model.get_layer("model").summary())
            #     print(model.summary())
            # model = NN_Clustering(N_rf, M, reduced_dim=8)
            # model = top_N_rf_user_model(M, K, N_rf)
            # model = partial_feedback_pure_greedy_model_not_perfect_CSI_available(N_rf, 32, 10, M, K, sigma2_n)
            # model = partial_feedback_pure_greedy_model(N_rf, 32, 1, M, K, sigma2_n)
            test_performance(model, M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h = sigma2_h)
            # test_DNN_different_K(model_path, M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h = sigma2_h)
            # vvvvvvvvvvvvvvvvvv using dynamic programming to do N_rf sweep of Greedy faster vvvvvvvvvvvvvvvvvv
            # ^^^^^^^^^^^^^^^^^^ using dynamic programming to do N_rf sweep of Greedy faster ^^^^^^^^^^^^^^^^^^
            # test_greedy(M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h = sigma2_h)
