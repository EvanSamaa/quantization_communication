import tensorflow as tf
from tf_agents.distributions import gumbel_softmax
from util import *
from models import *
import numpy as np
import scipy as sp
from relaxflow.reparam import CategoricalReparam
from relaxflow.relax import RELAX
from keras_adabound.optimizers import AdaBound
def grid_search_with_mutex_loss(N_rf = 8):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/Feb8th/user_loc0/on_user_loc_0_Nrf={}".format(N_rf)
    fname_template = fname_template_template + "{}"
    check = 250
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2

    # problem Definition
    M = 64
    K = 100
    seed = 100
    # N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0
    ############################### generate data ###############################
    valid_data = gen_realistic_data("trained_models/Feb8th/user_loc0/one_hundred_user_config_0.npy", 1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    ################################ hyperparameters ###############################
    EPOCHS = 100000
    lr = 0.0001
    N = 25 # number of
    rounds = 15
    sample_size = 50
    temp = 0.1
    check = 100
    model = FDD_agent_more_G(M, K, 5, N_rf, True, max_val)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    ################################ Metrics  ###############################
    sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    train_sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_hard_loss = tf.keras.metrics.Mean(name='train_loss')
    mutex_loss_fn = mutex_loss(N_rf, M, K, N)
    ################################ storing train data in npy file  ##############################
    # the three would be first train_loss, Hardloss, and the validation loss, every 50 iterations
    max_acc = 0
    np_data = ModelTrainer(save_dir=fname_template.format(".npy"), data_cols=3, epoch=EPOCHS)
    # training loop
    for i in range(0, EPOCHS):
        train_hard_loss.reset_states()
        # generate training data
        train_data = gen_realistic_data("trained_models/Feb8th/user_loc0/one_hundred_user_config_0.npy", N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        for e in range(0, rounds):
            temp = 0.5 * np.exp(-4.5 / rounds * e) * tf.maximum(0.0, ((200.0-i)/200.0)) + 0.1
            temp = np.float32(temp)
            train_hard_loss.reset_states()
            train_loss.reset_states()
            with tf.GradientTape(persistent=True) as tape:
                ###################### model post-processing ######################
                ans, raw_ans = model(train_data) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                out_raw = tf.transpose(raw_ans, [0, 1, 3, 2])
                out_raw = tf.reshape(out_raw[:,-1], [N * N_rf, K*M])
                sm = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw)
                out_raw = sm.sample(sample_size)
                out_raw = tf.reshape(out_raw, [sample_size, N, N_rf, K*M])
                out = tf.reduce_sum(out_raw, axis=2)
                out = tf.reshape(out, [sample_size*N, K*M])
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [sample_size,1, 1, 1]), [sample_size*N, K, M])
                ###################### model post-processing ######################
                loss = train_sum_rate(out, train_label) + mutex_loss_fn(raw_ans[:, -1])
                # loss = train_sum_rate(out, train_label) + 0.01 *mutex_loss_fn(out)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            # optimizer.minimize(loss, ans)
            train_loss(loss)
            train_hard_loss(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(ans[:,-1]), train_data))
            print(train_hard_loss.result(),train_loss.result())
            del tape
        ###################### testing with validation set ######################
        if i%check == 0:
            scheduled_output, raw_output = model.predict(valid_data, batch_size=N)
            valid_loss = tf.reduce_mean(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(scheduled_output[:, -1]), valid_data))
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), valid_loss])
            print("============================================================\n")
            print(valid_loss)
            if valid_loss < max_acc:
                max_acc = valid_loss
                model.save(fname_template.format(".h5"))
            if i >= (check * 2):
                graphing_data = np_data.data
                improvement = graphing_data[i + 1 - (check * 2): i - check + 1, -1].min() - graphing_data[
                                                                                                   i - check + 1: i + 1,
                                                                                                   -1].min()
                counter = 0
                for asldk in graphing_data[0:i + 1, -1]:
                    if asldk != 0:
                        print(counter, asldk)
                    counter = counter + 1
                print("the improvement in the past 500 epochs is: ", improvement)
                print("the validation SR is: ", valid_loss)
                if improvement <= 0.0001 and lr == 0.001:
                    lr = 0.0001
                    optimizer = tf.keras.optimizers.Adam(lr=0.0001)
                elif improvement <= 0.0001 and lr < 0.001:
                    break
        else:
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), 0])
    np_data.save()
def grid_search_with_mutex_loss_episodic(N_rf = 8):
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
                   "Per_link_Input_modification_most_G_raw_self_sigmoid":Per_link_Input_modification_most_G_raw_self_sigmoid,
                   "Per_link_Input_modification_most_G_raw_self_more_interference":Per_link_Input_modification_most_G_raw_self_more_interference,
                   "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum":Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum}
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/Feb8th/user_loc0/weighted_sumrate/weigh_inputs_Nrf={}".format(N_rf)
    fname_template = fname_template_template + "{}"
    check = 250
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2

    # problem Definition
    M = 64
    K = 100
    seed = 100
    # N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0
    ############################### generate data ###############################
    valid_data = gen_realistic_data("trained_models/Feb8th/user_loc0/one_hundred_user_config_0.npy", 1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    ################################ hyperparameters ###############################
    EPOCHS = 100000
    lr = 0.001
    N = 25 # number of
    rounds = 8
    sample_size = 20
    temp = 0.1
    check = 100
    episodes = 50
    alpha = .05
    # model = FDD_agent_more_G(M, K, 5, N_rf, True, max_val)
    model = tf.keras.models.load_model("trained_models/Feb8th/user_loc0/on_user_loc_0_Nrf={}.h5".format(N_rf), custom_objects=custome_obj)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    env = Weighted_sumrate_model(K, M, N_rf, N, alpha, hard_decision=True)
    ################################ Metrics  ###############################
    sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    # train_sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_hard_loss = tf.keras.metrics.Mean(name='train_loss')
    mutex_loss_fn = mutex_loss(N_rf, M, K, N)
    ################################ storing train data in npy file  ##############################
    # the three would be first train_loss, Hardloss, and the validation loss, every 50 iterations
    max_acc = 0
    np_data = ModelTrainer(save_dir=fname_template.format(".npy"), data_cols=3, epoch=EPOCHS)
    # training loop
    for i in range(0, EPOCHS):
        train_hard_loss.reset_states()
        # generate training data
        train_data = gen_realistic_data("trained_models/Feb8th/user_loc0/one_hundred_user_config_0.npy", N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        for e in range(0, rounds):
            env.reset()
            with tf.GradientTape(persistent=True) as tape:
                loss = 0
                gradients = 0
                for episode in range(episodes):
                    temp = 0.5 * np.exp(-4.5 / rounds * e) * tf.maximum(0.0, ((200.0-i)/200.0)) + 0.1
                    temp = np.float32(temp)
                    train_hard_loss.reset_states()
                    train_loss.reset_states()
                        ###################### model post-processing ######################
                    ans, raw_ans = model(train_data * tf.complex(tf.expand_dims(env.get_weight(), axis=2), 0.0)) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                    out_raw = tf.transpose(raw_ans, [0, 1, 3, 2])
                    out_raw = tf.reshape(out_raw[:,-1], [N * N_rf, K*M])
                    sm = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw)
                    out_raw = sm.sample(sample_size)
                    out_raw = tf.reshape(out_raw, [sample_size, N, N_rf, K*M])
                    out = tf.reduce_sum(out_raw, axis=2)
                    out = tf.reshape(out, [sample_size*N, K*M])
                    train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [sample_size,1, 1, 1]), [sample_size*N, K, M])
                    weight = tf.reshape(tf.tile(tf.expand_dims(env.get_weight(), axis=0), [sample_size,1, 1]), [sample_size*N, K])
                    ###################### model post-processing ######################
                    loss = env.compute_weighted_loss(out, train_label, weight=weight, update=False) + mutex_loss_fn(raw_ans[:, -1])
                    env.compute_weighted_loss(ans[:, -1], train_data, update=True)
                    env.increment()
                    print(tf.reduce_mean(tf.reduce_sum(env.rates, axis=2)))
                    # loss = train_sum_rate(out, train_label) + 0.01 *mutex_loss_fn(out)
                    gradients += tape.gradient(loss, model.trainable_variables)
                    del tape
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            # optimizer.minimize(loss, ans)
            train_loss(tf.reduce_mean(tf.reduce_sum(env.rates, axis=2)))
            train_hard_loss(tf.reduce_mean(tf.reduce_sum(env.rates, axis=2), axis=1)[0])
            print("\n===============overall=================\n",
                  train_hard_loss.result(),train_loss.result())
            del tape
        ###################### testing with validation set ######################
        if i%check == 0:
            scheduled_output, raw_output = model.predict(valid_data, batch_size=N)
            valid_loss = tf.reduce_mean(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(scheduled_output[:, -1]), valid_data))
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), valid_loss])
            print("============================================================\n")
            print(valid_loss)
            if valid_loss < max_acc:
                max_acc = valid_loss
                model.save(fname_template.format(".h5"))
            if i >= (check * 2):
                graphing_data = np_data.data
                improvement = graphing_data[i + 1 - (check * 2): i - check + 1, -1].min() - graphing_data[
                                                                                                   i - check + 1: i + 1,
                                                                                                   -1].min()
                counter = 0
                for asldk in graphing_data[0:i + 1, -1]:
                    if asldk != 0:
                        print(counter, asldk)
                    counter = counter + 1
                print("the improvement in the past 500 epochs is: ", improvement)
                print("the validation SR is: ", valid_loss)
                if improvement <= 0.0001 and lr == 0.001:
                    lr = 0.0001
                    optimizer = tf.keras.optimizers.Adam(lr=0.0001)
                elif improvement <= 0.0001 and lr < 0.001:
                    break
        else:
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), 0])
    np_data.save()
if __name__ == "__main__":
    M = 64
    K = 50
    seed = 100
    # N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0
    # np.random.seed(0)
    # gen_pathloss(1, 1, 100, 0.6, 0.1, 1, "trained_models/Feb8th/user_loc0/one_hundred_user_config_0.npy")
    # np.random.seed(1)
    # gen_pathloss(1, 1, 100, 0.6, 0.1, 1, "trained_models/Feb8th/one_hundred_user_config_1.npy")
    # np.random.seed(2)
    # gen_pathloss(1, 1, 100, 0.6, 0.1, 1, "trained_models/Feb8th/one_hundred_user_config_2.npy")
    for N_rf_to_search in [8]:
        grid_search_with_mutex_loss_episodic(N_rf_to_search)
