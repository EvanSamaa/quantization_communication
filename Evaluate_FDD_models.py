from util import *
from models import *
import numpy as np
# from scipy.io import savemat
import tensorflow as tf
# from matplotlib import pyplot as plt
def greedy_grid_search():
    M = 64
    K = 50
    sigma2_n = 1
    N_rf = 8
    sigma2_h = 0.0001
    out = np.zeros((64, 32, 8))
    for links in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
        for bits in range(1,33):
            if links*(6+bits) <= 128:
                model = DP_partial_feedback_pure_greedy_model(8, bits, links, M, K, sigma2_n, perfect_CSI=False)
                losses = test_greedy(model, M=M, K=K, B=bits, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h=sigma2_h, printing=False)
                out[links-1, bits-1, :] = losses
                print("{} links {} bits is done".format(links, bits))
            np.save("./trained_models/Dec_13/greedy_save_here/grid_search_all_under128.npy", out)


def test_greedy(model, M = 20, K = 5, B = 10, N_rf = 5, sigma2_h = 6.3, sigma2_n = 0.00001, printing=True):
    store=np.zeros((8,))
    num_data = 100
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # tp_fn = ExpectedThroughput(name = "throughput")
    result = np.zeros((3,))
    loss_fn1 = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    loss_fn2 = Total_activation_limit_hard(K, M, N_rf=0)
    print("Testing Starts")
    tf.random.set_seed(200)
    np.random.seed(200)
    ds_load = generate_link_channel_data(num_data, K, M, 1)
    prediction = model(ds_load)
    counter = 1
    for i in prediction:
        i_complex = tf.complex(tf.sqrt(counter*1.0), 0.0)
        out = loss_fn1(i, tf.abs(ds_load/i_complex))
        result[0] = tf.reduce_mean(out)
        result[1] = loss_fn2(i)
        if printing:
            print("the soft result is ", result)
            print("the variance is ", tf.math.reduce_std(out))
        store[counter-1] = tf.reduce_mean(out)
        counter = counter + 1
    return store
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

        valid_data = generate_link_channel_data(1000, K, M, Nrf=N_rf)
        garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
        q_train_data = tf.abs(ds_load) / max_val
        q_train_data = tf.where(q_train_data > 1.0, 1.0, q_train_data)
        q_train_data = tf.round(q_train_data * (2 ** 4 - 1)) / (2 ** 4 - 1) * max_val
        # prediction = ensumble_output(ds_load, model, k, loss_fn1) # this outputs (N, M*K, k)
        # prediction = model.predict(ds_load, batch_size=10)
        # prediction = model(ds_load)
        # compressed_G, position_matrix = G_compress(ds_load, 2)
        # scheduled_output, raw_output = model.predict_on_batch([ds_load, compressed_G, position_matrix])
        scheduled_output, raw_output, were = model.predict(q_train_data, batch_size=50)
        # scheduled_output, raw_output, input_mod, input_reconstructed_mod, reconstructed_input = model.predict_on_batch(ds_load)

        # scheduled_output, raw_output, recon = model(ds_load)

        # for i in range(0, num_data):
        # from matplotlib import pyplot as plt
        # for k in range(0, num_data):
        #     G_pred = DP_partial_feedback_pure_greedy_model(N_rf, 32, 10, M, K, sigma2_n, True)(ds_load[k:k+1])
        #     for i in range(0,5):
        #         prediction = scheduled_output[:, i]
        #         # plt.imshow(tf.reshape(prediction[k], (K, M)))
        #         plt.plot(np.arange(0, K*M), G_pred[-1][0])
        #         plt.plot(np.arange(0, K*M), prediction[k])
        #
        #         plt.show()
        # A[2]
        prediction = scheduled_output[:, -1]
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
    from matplotlib import pyplot as plt
    cut = 0
    for i in range(arr.shape[0]-1, 0, -1):
        if arr[i].any != 0:
            cut = i
            break
    arr = arr[:cut, :]
    x = np.arange(0, arr.shape[0])
    for i in col:
        plt.plot(x, arr[:, i])
    # plt.plot(x, arr[:, 3])
    plt.title(title)
    plt.show()
def garsons_method(model_path):
    from matplotlib import pyplot as plt
    model = tf.keras.models.load_model(model_path, custom_objects=custome_obj)
    dnn = model.get_layer("DNN_within_model0")
    weights = dnn.get_layer("Dense1_inside_DNN0")
    kernel = weights.kernel
    garson_importance = tf.reduce_sum(tf.abs(kernel), axis=1)
    norm = tf.reduce_sum(garson_importance, keepdims=True)
    garson_importance = tf.divide(garson_importance, norm)
    plt.plot(garson_importance.numpy(), '+')
    plt.show()
if __name__ == "__main__":
    greedy_grid_search()
    # Axes3D import has side effects, it enables using projection='3d' in add_subplot
    custome_obj = {'Closest_embedding_layer': Closest_embedding_layer, 'Interference_Input_modification': Interference_Input_modification,
                   'Interference_Input_modification_no_loop': Interference_Input_modification_no_loop,
                   "Interference_Input_modification_per_user":Interference_Input_modification_per_user,
                   "Closest_embedding_layer_moving_avg":Closest_embedding_layer_moving_avg,
                   "Per_link_Input_modification_more_G":Per_link_Input_modification_more_G,
                   "Per_link_Input_modification_more_G_less_X":Per_link_Input_modification_more_G_less_X,
                   "Per_link_Input_modification_even_more_G":Per_link_Input_modification_even_more_G,
                   "Per_link_Input_modification_compress_XG":Per_link_Input_modification_compress_XG,
                   "Per_link_Input_modification_compress_XG_alt": Per_link_Input_modification_compress_XG_alt,
                   "Per_link_Input_modification_more_G_alt_2":Per_link_Input_modification_more_G_alt_2,
                   "Per_link_Input_modification_compress_XG_alt_2":Per_link_Input_modification_compress_XG_alt_2,
                   "Per_link_Input_modification_most_G":Per_link_Input_modification_most_G,
                   "Per_link_sequential_modification": Per_link_sequential_modification,
                   "Per_link_sequential_modification_compressedX":Per_link_sequential_modification_compressedX,
                   "Per_link_Input_modification_most_G_raw_self":Per_link_Input_modification_most_G_raw_self,
                   "Reduced_output_input_mod":Reduced_output_input_mod,
                   "TopPrecoderPerUserInputMod":TopPrecoderPerUserInputMod,
                   "X_extends":X_extends,
                   "Per_link_Input_modification_most_G_col":Per_link_Input_modification_most_G_col,
                   "Sparsemax":Sparsemax,
                   "Sequential_Per_link_Input_modification_most_G_raw_self":Sequential_Per_link_Input_modification_most_G_raw_self,
                   "Per_link_Input_modification_most_G_raw_self_sigmoid":Per_link_Input_modification_most_G_raw_self_sigmoid}
    # training_data = np.load("trained_models\Dec_13\GNN_grid_search_temp=0.1.npy")
    # plot_data(training_data, [2], "sum rate")
    file = "trained_models\Dec_13\with_feedback\it32\GNN_annealing_temp_Nrf={}+limit_res"
    # file = "trained_models/Nov_23/B=32_one_CE_loss/N_rf=1+VAEB=1x32E=4+1x512_per_linkx6_alt+CE_loss+MP"
    # for item in [0.01, 0.1, 1, 5, 10]:
    #     garsons_method(file.format(item))
    # obtain_channel_distributions(10000, 50, 64, 5)
    # A[2]
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
    mores = [8,7,6,5,4,3,2,1]
    Es = [1]
    # model = DP_partial_feedback_pure_greedy_model(8, 2, 2, M, K, sigma2_n, perfect_CSI=False)
    # test_greedy(model, M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h=sigma2_h)
    # model = DP_partial_feedback_pure_greedy_model(8, 2, 5, M, K, sigma2_n, perfect_CSI=False)
    # test_greedy(model, M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h=sigma2_h)
    for j in Es:
        for i in mores:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            N_rf = i
            bits=j
            print("========================================== lambda =", j, "Nrf = ", i)

            model = tf.keras.models.load_model(model_path.format(N_rf), custom_objects=custome_obj)
            # print(model.get_layer("model_2").get_layer("model_1").summary())
            # model = tf.keras.models.load_model(model_path, custom_objects=custome_obj)
            # model = partial_feedback_top_N_rf_model(N_rf, B, 1, M, K, sigma2_n)
            #     print(model.get_layer("model").summary())
            #     print(model.summary())
            # model = NN_Clustering(N_rf, M, reduced_dim=8)
            # model = top_N_rf_user_model(M, K, N_rf)
            # model = partial_feedback_pure_greedy_model_not_perfect_CSI_available(N_rf, 32, 10, M, K, sigma2_n)
            # model = partial_feedback_pure_greedy_model(N_rf, 32, i, M, K, sigma2_n)
            # model = relaxation_based_solver(M, K, N_rf)
            test_performance(model, M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h = sigma2_h)
            # test_DNN_different_K(model_path, M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h = sigma2_h)
            # vvvvvvvvvvvvvvvvvv using dynamic programming to do N_rf sweep of Greedy faster vvvvvvvvvvvvvvvvvv
            # ^^^^^^^^^^^^^^^^^^ using dynamic programming to do N_rf sweep of Greedy faster ^^^^^^^^^^^^^^^^^^
            # test_greedy(M=M, K=K, B=B, N_rf=N_rf, sigma2_n=sigma2_n, sigma2_h = sigma2_h)
