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
    fname_template_template = "trained_models/Feb8th/user_loc0/weighted_sumrate/binary_weighted_inputs_Nrf={}".format(N_rf)
    fname_template = fname_template_template + "{}"
    check = 30
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
    rounds = 8
    sample_size = 20
    temp = 0.1
    check = 100
    episodes = 50
    alpha = .3
    # model = FDD_agent_more_G(M, K, 5, N_rf, True, max_val)
    model = tf.keras.models.load_model("trained_models/Feb8th/user_loc0/on_user_loc_0_Nrf={}.h5".format(N_rf), custom_objects=custome_obj)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    env = Weighted_sumrate_model(K, M, N_rf, N, alpha, hard_decision=False)
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
            gradients = []
            for episode in range(episodes):
                with tf.GradientTape(persistent=True) as tape:
                    temp = 0.5 * np.exp(-4.5 / rounds * e) * tf.maximum(0.0, ((200.0-i)/200.0)) + 0.1
                    temp = np.float32(temp)
                    train_hard_loss.reset_states()
                    train_loss.reset_states()
                        ###################### model post-processing ######################
                    ans, raw_ans = model(train_data * tf.complex(tf.expand_dims(env.get_binary_weights(), axis=2), 0.0)) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                    out_raw = tf.transpose(raw_ans, [0, 1, 3, 2])
                    out_raw = tf.reshape(out_raw[:,-1], [N * N_rf, K*M])
                    sm = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw)
                    out_raw = sm.sample(sample_size)
                    out_raw = tf.reshape(out_raw, [sample_size, N, N_rf, K*M])
                    out = tf.reduce_sum(out_raw, axis=2)
                    out = tf.reshape(out, [sample_size*N, K*M])
                    train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [sample_size,1, 1, 1]), [sample_size*N, K, M])
                    weight = tf.reshape(tf.tile(tf.expand_dims(env.get_binary_weights(), axis=0), [sample_size,1, 1]), [sample_size*N, K])
                    ###################### model post-processing ######################
                    loss = env.compute_weighted_loss(out, train_label, weight=weight, update=False) + mutex_loss_fn(raw_ans[:, -1])
                    env.compute_weighted_loss(ans[:, -1], train_data, update=True)
                    env.increment()
                print(tf.reduce_mean(tf.reduce_sum(env.rates, axis=2)))
                # loss = train_sum_rate(out, train_label) + 0.01 *mutex_loss_fn(out)
                curr = tape.gradient(loss, model.trainable_variables)
                if len(gradients) == 0:
                    for i in range(0, len(curr)):
                        gradients += [curr[i]/episodes]
                else:
                    for i in range(0, len(curr)):
                        gradients[i] += curr[i]/episodes
                del tape

            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            # optimizer.minimize(loss, ans)
            l1 = tf.reduce_mean(tf.reduce_sum(env.rates, axis=2))
            lh = tf.reduce_mean(tf.reduce_sum(env.rates, axis=2), axis=1)[0]
            train_loss(l1)
            train_hard_loss(lh)
            print("\n===============overall=================\n",
                  l1,lh)
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
def grid_search_with_mutex_loss_episodic_new_archi(N_rf = 8):
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
                   "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum":Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum,
                   "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights":Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights}
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/Feb8th/user_loc0/weighted_sumrate/prior_weight_then_RF_training={}".format(N_rf)
    fname_template = fname_template_template + "{}"
    # problem Definition
    pre_train = 0
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
    alpha = .95
    model = FDD_agent_more_G_with_weights(M, K, 5, N_rf, True, max_val)
    # model = tf.keras.models.load_model("trained_models/Feb8th/user_loc0/on_user_loc_0_Nrf={}.h5".format(N_rf), custom_objects=custome_obj)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    env = Weighted_sumrate_model(K, M, N_rf, N, alpha, hard_decision=False)
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
        if i >= pre_train:
            ###################### training happens here ######################
            check = 30
            for e in range(0, rounds):
                env.reset()
                for episode in range(episodes):
                    with tf.GradientTape(persistent=True) as tape:
                        temp = 0.5 * np.exp(-4.5 / rounds * e) * tf.maximum(0.0, ((200.0-i)/200.0)) + 0.1
                        temp = np.float32(temp)
                        train_hard_loss.reset_states()
                        train_loss.reset_states()
                        ###################### model post-processing ######################
                        input_mod = tf.concat([train_data, tf.complex(tf.expand_dims(env.get_weight(), axis=2), 0.0)], axis = 2)
                        ans, raw_ans = model(input_mod) # raw_ans is in the shape of (N, passes, M*K, N_rf)
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
                        env.increment()
                        env.compute_weighted_loss(ans[:, -1], train_data, update=True)
                    print(tf.reduce_mean(tf.reduce_sum(env.rates, axis=2)))
                    # loss = train_sum_rate(out, train_label) + 0.01 *mutex_loss_fn(out)
                    curr = tape.gradient(loss, model.trainable_variables)
                    if len(gradients) == 0:
                        for i in range(0, len(curr)):
                            gradients += [curr[i]/episodes]
                    else:
                        for i in range(0, len(curr)):
                            gradients[i] += curr[i]/episodes
                    del tape

                optimizer.apply_gradients(zip(gradients,model.trainable_variables))
                # optimizer.minimize(loss, ans)
                l1 = tf.reduce_mean(tf.reduce_sum(env.rates, axis=2))
                lh = tf.reduce_mean(tf.reduce_sum(env.rates, axis=2), axis=1)[0]
                train_loss(l1)
                train_hard_loss(lh)
                print("\n===============overall=================\n",
                      l1,lh)
        else:
            train_hard_loss.reset_states()
            # generate training data
            ###################### training happens here ######################
            for e in range(0, rounds):
                with tf.GradientTape(persistent=True) as tape:
                    temp = 0.5 * np.exp(-4.5 / rounds * e) * tf.maximum(0.0, ((200.0 - i) / 200.0)) + 0.1
                    temp = np.float32(temp)
                    train_hard_loss.reset_states()
                    train_loss.reset_states()
                    ###################### model post-processing ######################
                    input_mod = tf.concat([train_data, tf.complex(tf.ones([N, K, 1], dtype=tf.float32), 0.0)],
                                          axis=2)
                    ans, raw_ans = model(input_mod)  # raw_ans is in the shape of (N, passes, M*K, N_rf)
                    out_raw = tf.transpose(raw_ans, [0, 1, 3, 2])
                    out_raw = tf.reshape(out_raw[:, -1], [N * N_rf, K * M])
                    sm = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw)
                    out_raw = sm.sample(sample_size)
                    out_raw = tf.reshape(out_raw, [sample_size, N, N_rf, K * M])
                    out = tf.reduce_sum(out_raw, axis=2)
                    out = tf.reshape(out, [sample_size * N, K * M])
                    train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [sample_size, 1, 1, 1]),
                                             [sample_size * N, K, M])
                    ###################### model post-processing ######################
                    # loss = train_label(out, train_label) + mutex_loss_fn(raw_ans[:, -1])
                    loss = sum_rate(out, train_label) + 0.01*mutex_loss_fn(raw_ans[:, -1])
                gradients = tape.gradient(loss, model.trainable_variables)

                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                train_loss(loss)
                train_hard_loss(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(ans[:, -1]), train_data))
                print(train_hard_loss.result(), train_loss.result())
        ###################### testing with validation set ######################
        if i%check == 0:
            input_mod=tf.concat([valid_data, tf.complex(tf.ones([valid_data.shape[0], K, 1], dtype=tf.float32), 0.0)],
                                          axis=2)
            scheduled_output, raw_output = model.predict(input_mod, batch_size=N)
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
                elif improvement <= 0.0001 and lr < 0.001 and i > pre_train:
                    break
        else:
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), 0])
    np_data.save()
def grid_search_with_mutex_loss_weighted_sumrate_train_as_if_non_episodic(N_rf = 8):
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
                   "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum":Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum,
                   "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights":Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights}
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/Feb8th/user_loc0/as_if_non_episodic/alpha=0p95_NRF={}_fixed_loss".format(N_rf)
    fname_template = fname_template_template + "{}"
    # problem Definition
    pre_train = 0
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
    N = 15 # number of
    rounds = 8
    sample_size = 15
    temp = 0.1
    check = 40
    episodes = 50
    alpha = .95
    model = FDD_agent_more_G_with_weights(M, K, 5, N_rf, True, max_val)
    # model = tf.keras.models.load_model("trained_models/Feb8th/user_loc0/on_user_loc_0_Nrf={}.h5".format(N_rf), custom_objects=custome_obj)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    env = Weighted_sumrate_model(K, M, N_rf, N, alpha, hard_decision=False, loss_fn=Sum_rate_utility_WeiCui_seperate_user_stable(K, M, 1))
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
        env.reset()
        ###################### testing with validation set ######################
        for episode in range(episodes):
        # for episode in range(0):
            env.increment()
            for e in range(0, rounds):
                with tf.GradientTape(persistent=True) as tape:
                    temp = 0.5 * np.exp(-4.5 / rounds * e) * tf.maximum(0.0, ((200.0-i)/200.0)) + 0.1
                    temp = np.float32(temp)
                    train_hard_loss.reset_states()
                    train_loss.reset_states()
                    ###################### model post-processing ######################
                    input_mod = tf.concat([train_data, tf.complex(tf.expand_dims(env.get_weight(), axis=2), 0.0)], axis = 2)
                    ans, raw_ans = model(input_mod) # raw_ans is in the shape of (N, passes, M*K, N_rf)
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
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                del tape
            print(tf.reduce_mean(tf.reduce_sum(env.get_rates(), axis=2)))
                # loss = train_sum_rate(out, train_label) + 0.01 *mutex_loss_fn(out)
            # optimizer.minimize(loss, ans)
            l1 = tf.reduce_mean(tf.reduce_sum(env.get_rates(), axis=2))
            lh = tf.reduce_mean(tf.reduce_sum(env.get_rates(), axis=2), axis=1)[0]
            train_loss(l1)
            train_hard_loss(lh)
            print("\n===============overall=================\n",
                  l1,lh)
        if i%check == 0:
            input_mod=tf.concat([valid_data, tf.complex(tf.ones([valid_data.shape[0], K, 1], dtype=tf.float32), 0.0)],
                                axis=2)
            scheduled_output = []
            raw_output = []
            for batches in range(0, math.floor(valid_data.shape[0]/10)):
                scheduled_output_temp, raw_output_temp = model(input_mod[batches * 10:(batches + 1) * 10, :, :])
                scheduled_output.append(scheduled_output_temp)
                raw_output.append(raw_output_temp)
            scheduled_output = tf.concat(scheduled_output, axis = 0)
            raw_output = tf.concat(raw_output, axis = 0)
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
                    optimizer = tf.keras.optimizers.Adam(lr=lr)
                elif improvement <= 0.0001 and lr < 0.001 and i > pre_train:
                    break
        else:
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), 0])
    np_data.save()
def grid_search_with_mutex_loss_weighted_sumrate_train_random_0_1(K, N_rf = 8):
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
                   "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum":Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum,
                   "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights":Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights}
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/Apr5th/K20/dnn_with_01_weight/random_binary_NRF={}_biggerbatch_penalize_small_weight".format(N_rf)
    positional_config = "trained_models/Apr5th/K20/user_positions.npy"
    fname_template = fname_template_template + "{}"
    # problem Definition
    pre_train = 0
    M = 64
    seed = 100
    # N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0
    ############################### generate data ###############################
    valid_data = gen_realistic_data(positional_config, 1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    ################################ hyperparameters ###############################
    EPOCHS = 100000
    lr = 0.001
    N = 50 # number of
    rounds = 8
    sample_size = 10
    temp = 0.1
    check = 200
    # episodes = 50
    alpha = .05
    model = FDD_agent_more_G_with_weights(M, K, 5, N_rf, True, max_val)
    # model = tf.keras.models.load_model("trained_models/Feb8th/user_loc0/on_user_loc_0_Nrf={}.h5".format(N_rf), custom_objects=custome_obj)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    # env = Weighted_sumrate_model(K, M, N_rf, N, alpha, hard_decision=False, loss_fn=Sum_rate_utility_WeiCui_seperate_user_stable(K, M, 1))
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
    env = Weighted_sumrate_model(K, M, N_rf, N, alpha, hard_decision=False)
    for i in range(0, EPOCHS):
        train_hard_loss.reset_states()
        # generate training data
        train_data = gen_realistic_data(positional_config, N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        ###################### testing with validation set ######################
        weight_indices = []
        current_iter_up_nums = np.random.randint(10, K)
        for h in range(0, N):
           weight_indices.append(np.random.choice(K, (1, current_iter_up_nums), replace=False))
        weight_indices = np.concatenate(weight_indices, axis=0)
        current_weights = tf.one_hot(weight_indices, K)
        current_weights = tf.reduce_sum(current_weights, axis=1)
        for e in range(0, rounds):
            with tf.GradientTape(persistent=True) as tape:
                temp = 0.5 * np.exp(-4.5 / rounds * e) * tf.maximum(0.0, ((200.0-i)/200.0)) + 0.1
                temp = np.float32(temp)
                train_hard_loss.reset_states()
                train_loss.reset_states()
                ###################### model post-processing ######################
                input_mod = tf.concat([train_data, tf.complex(tf.expand_dims(current_weights, axis=2), 0.0)], axis = 2)
                ans, raw_ans = model(input_mod) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                out_raw = tf.transpose(raw_ans, [0, 1, 3, 2])
                out_raw = tf.reshape(out_raw[:,-1], [N * N_rf, K*M])
                sm = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw)
                out_raw = sm.sample(sample_size)
                out_raw = tf.reshape(out_raw, [sample_size, N, N_rf, K*M])
                out = tf.reduce_sum(out_raw, axis=2)
                out = tf.reshape(out, [sample_size*N, K*M])
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [sample_size,1, 1, 1]), [sample_size*N, K, M])
                weight = tf.reshape(tf.tile(tf.expand_dims(current_weights, axis=0), [sample_size,1, 1]), [sample_size*N, K])
                ###################### model post-processing ######################
                weight_sr = env.compute_weighted_loss(out, train_label, weight=weight, update=False)
                tiled_weight = tf.reshape(tf.tile(tf.expand_dims(weight, axis=2), (1, 1, M)), (sample_size*N, K*M))
                on_off_loss = tf.reduce_sum(tf.square(tiled_weight - out) * (1.0 - tiled_weight), axis=1)
                on_off_loss = tf.reduce_mean(on_off_loss)
                loss = weight_sr + mutex_loss_fn(raw_ans[:, -1]) + on_off_loss
                # loss = env.compute_weighted_loss(ans[:, -1], train_data, update=True, weight=current_weights) + mutex_loss_fn(raw_ans[:, -1])
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            del tape
                # loss = train_sum_rate(out, train_label) + 0.01 *mutex_loss_fn(out)
            # optimizer.minimize(loss, ans)
            # l1 = tf.reduce_mean(tf.reduce_sum(env.get_weighted_rates(), axis=2))
            # lh = tf.reduce_mean(tf.reduce_sum(env.get_weighted_rates(), axis=2), axis=1)[0]
            loss_hard = env.compute_weighted_loss(Harden_scheduling_user_constrained(N_rf, K, M)(ans[:,-1]), train_data, weight=current_weights, update=False)
            train_loss(weight_sr)
            train_hard_loss(loss_hard)
            print("soft result: ", train_loss.result())
            print("hard result: ", train_hard_loss.result())
            print("on-off loss: ", on_off_loss)
        if i%check == 0:
            input_mod=tf.concat([valid_data, tf.complex(tf.ones([valid_data.shape[0], K, 1], dtype=tf.float32), 0.0)],
                                axis=2)
            scheduled_output = []
            raw_output = []
            for batches in range(0, math.floor(valid_data.shape[0]/10)):
                scheduled_output_temp, raw_output_temp = model(input_mod[batches * 10:(batches + 1) * 10, :, :])
                scheduled_output.append(scheduled_output_temp)
                raw_output.append(raw_output_temp)
            scheduled_output = tf.concat(scheduled_output, axis = 0)
            raw_output = tf.concat(raw_output, axis = 0)
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
                    optimizer = tf.keras.optimizers.Adam(lr=lr)
                elif improvement <= 0.0001 and lr < 0.001 and i > pre_train:
                    break
        else:
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), 0])
    np_data.save()
def grid_search_with_mutex_loss_weighted_sumrate_train_normal0_1(K, N_rf = 8):
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
                   "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum":Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum,
                   "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights":Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights}
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/Apr5th/K20/train_dnn_with_random_normal_weight/random_binary_NRF={}_biggerbatch_penalize_small_weight".format(N_rf)
    positional_config = "trained_models/Apr5th/K20/user_positions.npy"
    fname_template = fname_template_template + "{}"
    # problem Definition
    pre_train = 0
    M = 64
    seed = 100
    # N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0
    ############################### generate data ###############################
    valid_data = gen_realistic_data(positional_config, 1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    ################################ hyperparameters ###############################
    EPOCHS = 100000
    lr = 0.001
    N = 50 # number of
    rounds = 8
    sample_size = 10
    temp = 0.1
    check = 200
    # episodes = 50
    alpha = .05
    model = FDD_agent_more_G_with_weights(M, K, 5, N_rf, True, max_val)
    # model = tf.keras.models.load_model("trained_models/Feb8th/user_loc0/on_user_loc_0_Nrf={}.h5".format(N_rf), custom_objects=custome_obj)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    # env = Weighted_sumrate_model(K, M, N_rf, N, alpha, hard_decision=False, loss_fn=Sum_rate_utility_WeiCui_seperate_user_stable(K, M, 1))
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
    env = Weighted_sumrate_model(K, M, N_rf, N, alpha, hard_decision=False)
    for i in range(0, EPOCHS):
        train_hard_loss.reset_states()
        # generate training data
        train_data = gen_realistic_data(positional_config, N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        ###################### testing with validation set ######################
        current_weights = tf.random.truncated_normal((N, K), 0.5, 0.25)
        for e in range(0, rounds):
            with tf.GradientTape(persistent=True) as tape:
                temp = 0.5 * np.exp(-4.5 / rounds * e) * tf.maximum(0.0, ((200.0-i)/200.0)) + 0.1
                temp = np.float32(temp)
                train_hard_loss.reset_states()
                train_loss.reset_states()
                ###################### model post-processing ######################
                input_mod = tf.concat([train_data, tf.complex(tf.expand_dims(current_weights, axis=2), 0.0)], axis = 2)
                ans, raw_ans = model(input_mod) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                out_raw = tf.transpose(raw_ans, [0, 1, 3, 2])
                out_raw = tf.reshape(out_raw[:,-1], [N * N_rf, K*M])
                sm = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw)
                out_raw = sm.sample(sample_size)
                out_raw = tf.reshape(out_raw, [sample_size, N, N_rf, K*M])
                out = tf.reduce_sum(out_raw, axis=2)
                out = tf.reshape(out, [sample_size*N, K*M])
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [sample_size,1, 1, 1]), [sample_size*N, K, M])
                weight = tf.reshape(tf.tile(tf.expand_dims(current_weights, axis=0), [sample_size,1, 1]), [sample_size*N, K])
                ###################### model post-processing ######################
                weight_sr = env.compute_weighted_loss(out, train_label, weight=2*(weight-0.5), update=False)
                loss = weight_sr + mutex_loss_fn(raw_ans[:, -1])
                # loss = env.compute_weighted_loss(ans[:, -1], train_data, update=True, weight=current_weights) + mutex_loss_fn(raw_ans[:, -1])
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            del tape
                # loss = train_sum_rate(out, train_label) + 0.01 *mutex_loss_fn(out)
            # optimizer.minimize(loss, ans)
            # l1 = tf.reduce_mean(tf.reduce_sum(env.get_weighted_rates(), axis=2))
            # lh = tf.reduce_mean(tf.reduce_sum(env.get_weighted_rates(), axis=2), axis=1)[0]
            loss_hard = env.compute_weighted_loss(Harden_scheduling_user_constrained(N_rf, K, M)(ans[:,-1]), train_data, weight=current_weights, update=False)
            train_loss(weight_sr)
            train_hard_loss(loss_hard)
            print("soft result: ", train_loss.result(), "hard result: ", train_hard_loss.result())
        if i%check == 0:
            input_mod=tf.concat([valid_data, tf.complex(tf.random.truncated_normal((1000, K, 1), 0.5, 0.25), 0.0)],
                                axis=2)
            scheduled_output = []
            raw_output = []
            for batches in range(0, math.floor(valid_data.shape[0]/10)):
                scheduled_output_temp, raw_output_temp = model(input_mod[batches * 10:(batches + 1) * 10, :, :])
                scheduled_output.append(scheduled_output_temp)
                raw_output.append(raw_output_temp)
            scheduled_output = tf.concat(scheduled_output, axis = 0)
            raw_output = tf.concat(raw_output, axis = 0)
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
                    optimizer = tf.keras.optimizers.Adam(lr=lr)
                elif improvement <= 0.0001 and lr < 0.001 and i > pre_train:
                    break
        else:
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), 0])
    np_data.save()
def grid_search_with_mutex_loss_weighted_sumrate_train_normal0_1_duel_network(K, N_rf = 8):
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
                   "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum":Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum,
                   "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights":Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights}
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/Apr5th/K20/duel_dnn_with_random_normal_weight/random_binary_NRF={}_biggerbatch_penalize_small_weight".format(N_rf)
    positional_config = "trained_models/Apr5th/K20/user_positions.npy"
    fname_template = fname_template_template + "{}"
    # problem Definition
    pre_train = 0
    M = 64
    seed = 100
    # N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0
    ############################### generate data ###############################
    valid_data = gen_realistic_data(positional_config, 1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    ################################ hyperparameters ###############################
    EPOCHS = 100000
    lr = 0.001
    N = 50 # number of
    rounds = 8
    sample_size = 10
    temp = 0.1
    check = 200
    # episodes = 50
    alpha = .05
    # model = FDD_agent_more_G_with_weights(M, K, 5, N_rf, True, max_val)
    model = FDD_agent_more_G_duel_network(M, K, 5, N_rf, True, max_val)
    # model = tf.keras.models.load_model("trained_models/Feb8th/user_loc0/on_user_loc_0_Nrf={}.h5".format(N_rf), custom_objects=custome_obj)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    # env = Weighted_sumrate_model(K, M, N_rf, N, alpha, hard_decision=False, loss_fn=Sum_rate_utility_WeiCui_seperate_user_stable(K, M, 1))
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
    env = Weighted_sumrate_model(K, M, N_rf, N, alpha, hard_decision=False)
    for i in range(0, EPOCHS):
        train_hard_loss.reset_states()
        # generate training data
        train_data = gen_realistic_data(positional_config, N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        ###################### testing with validation set ######################
        current_weights = tf.random.uniform((N, K), 0.0000001, 1)
        for e in range(0, rounds):
            with tf.GradientTape(persistent=True) as tape:
                temp = 0.5 * np.exp(-4.5 / rounds * e) * tf.maximum(0.0, ((200.0-i)/200.0)) + 0.1
                temp = np.float32(temp)
                train_hard_loss.reset_states()
                train_loss.reset_states()
                ###################### model post-processing ######################
                input_mod = tf.concat([train_data, tf.complex(tf.expand_dims(current_weights, axis=2), 0.0)], axis = 2)
                ans, raw_ans, activations = model(input_mod) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                out_raw = tf.transpose(raw_ans, [0, 1, 3, 2])
                out_raw = tf.reshape(out_raw[:,-1], [N * N_rf, K*M])
                activations = tf.reshape(tf.tile(tf.expand_dims(activations, axis=0), [N_rf, 1, 1]),[N_rf * N, K*M])
                out_raw = out_raw * activations
                sm = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw)
                out_raw = sm.sample(sample_size)
                out_raw = tf.reshape(out_raw, [sample_size, N, N_rf, K*M])
                out = tf.reduce_sum(out_raw, axis=2)
                out = tf.reshape(out, [sample_size*N, K*M])
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [sample_size,1, 1, 1]), [sample_size*N, K, M])
                weight = tf.reshape(tf.tile(tf.expand_dims(current_weights, axis=0), [sample_size,1, 1]), [sample_size*N, K])

                ###################### model post-processing ######################
                weight_sr = env.compute_weighted_loss(out, train_label, weight=weight, update=False)
                loss = weight_sr + mutex_loss_fn(raw_ans[:, -1])
                # loss = env.compute_weighted_loss(ans[:, -1], train_data, update=True, weight=current_weights) + mutex_loss_fn(raw_ans[:, -1])
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            del tape
                # loss = train_sum_rate(out, train_label) + 0.01 *mutex_loss_fn(out)
            # optimizer.minimize(loss, ans)
            # l1 = tf.reduce_mean(tf.reduce_sum(env.get_weighted_rates(), axis=2))
            # lh = tf.reduce_mean(tf.reduce_sum(env.get_weighted_rates(), axis=2), axis=1)[0]
            loss_hard = env.compute_weighted_loss(Harden_scheduling_user_constrained(N_rf, K, M)(ans[:,-1]), train_data, weight=current_weights, update=False)
            train_loss(weight_sr)
            train_hard_loss(loss_hard)
            print("soft result: ", train_loss.result(), "hard result: ", train_hard_loss.result())
        if i%check == 0:
            input_mod=tf.concat([valid_data, tf.complex(tf.random.truncated_normal((1000, K, 1), 0.5, 0.25), 0.0)],
                                axis=2)
            scheduled_output = []
            raw_output = []
            for batches in range(0, math.floor(valid_data.shape[0]/10)):
                scheduled_output_temp, raw_output_temp, _garb = model(input_mod[batches * 10:(batches + 1) * 10, :, :])
                scheduled_output.append(scheduled_output_temp)
                raw_output.append(raw_output_temp)
            scheduled_output = tf.concat(scheduled_output, axis = 0)
            raw_output = tf.concat(raw_output, axis = 0)
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
                    optimizer = tf.keras.optimizers.Adam(lr=lr)
                elif improvement <= 0.0001 and lr < 0.001 and i > pre_train:
                    break
        else:
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), 0])
    np_data.save()
def grid_search_with_mutex_loss_weighted_sumrate_train_random_0_1_modify_gain(K, N_rf = 8):
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
                   "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum":Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum,
                   "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights":Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights}
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/Apr5th/K{}/train_with_0_1_weight_withOldModel/random_binary_NRF={}_biggerbatch".format(K, N_rf)
    positional_config = "trained_models/Apr5th/K{}/user_positions.npy".format(K)
    fname_template = fname_template_template + "{}"
    # problem Definition
    pre_train = 0
    M = 64
    # K = 100
    seed = 100
    # N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0
    ############################### generate data ###############################
    valid_data = gen_realistic_data(positional_config, 1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    ################################ hyperparameters ###############################
    EPOCHS = 100000
    lr = 0.001
    N = 50 # number of
    rounds = 8
    sample_size = 10
    temp = 0.1
    check = 200
    # episodes = 50
    alpha = .05
    # model = FDD_agent_more_G_with_weights(M, K, 5, N_rf, True, max_val)
    model = FDD_agent_more_G(M, K, 5, N_rf, True, max_val)
    # model = tf.keras.models.load_model("trained_models/Feb8th/user_loc0/on_user_loc_0_Nrf={}.h5".format(N_rf), custom_objects=custome_obj)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    # env = Weighted_sumrate_model(K, M, N_rf, N, alpha, hard_decision=False, loss_fn=Sum_rate_utility_WeiCui_seperate_user_stable(K, M, 1))
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
    env = Weighted_sumrate_model(K, M, N_rf, N, alpha, hard_decision=False)
    for i in range(0, EPOCHS):
        train_hard_loss.reset_states()
        # generate training data
        train_data = gen_realistic_data(positional_config, N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        ###################### testing with validation set ######################
        weight_indices = []
        current_iter_up_nums = np.random.randint(10, K)
        for h in range(0, N):
           weight_indices.append(np.random.choice(K, (1, current_iter_up_nums), replace=False))
        weight_indices = np.concatenate(weight_indices, axis=0)
        current_weights = tf.one_hot(weight_indices, K)
        current_weights = tf.reduce_sum(current_weights, axis=1)
        for e in range(0, rounds):
            with tf.GradientTape(persistent=True) as tape:
                temp = 0.5 * np.exp(-4.5 / rounds * e) * tf.maximum(0.0, ((200.0-i)/200.0)) + 0.1
                temp = np.float32(temp)
                train_hard_loss.reset_states()
                train_loss.reset_states()
                ###################### model post-processing ######################
                # input_mod = tf.concat([train_data, tf.complex(tf.expand_dims(current_weights, axis=2), 0.0)], axis = 2)
                input_mod = train_data * tf.complex(tf.expand_dims(current_weights, axis=2), 0.0)
                ans, raw_ans = model(input_mod) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                out_raw = tf.transpose(raw_ans, [0, 1, 3, 2])
                out_raw = tf.reshape(out_raw[:,-1], [N * N_rf, K*M])
                sm = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw)
                out_raw = sm.sample(sample_size)
                out_raw = tf.reshape(out_raw, [sample_size, N, N_rf, K*M])
                out = tf.reduce_sum(out_raw, axis=2)
                out = tf.reshape(out, [sample_size*N, K*M])
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [sample_size,1, 1, 1]), [sample_size*N, K, M])
                weight = tf.reshape(tf.tile(tf.expand_dims(current_weights, axis=0), [sample_size,1, 1]), [sample_size*N, K])
                ###################### model post-processing ######################
                weight_sr = env.compute_weighted_loss(out, train_label, weight=weight, update=False)
                tiled_weight = tf.reshape(tf.tile(tf.expand_dims(weight, axis=2), (1, 1, M)), (sample_size*N, K*M))
                on_off_loss = tf.reduce_sum(tf.square(tiled_weight - out) * (1.0 - tiled_weight), axis=1)
                on_off_loss = tf.reduce_mean(on_off_loss)
                loss = weight_sr + mutex_loss_fn(raw_ans[:, -1]) + on_off_loss
                # loss = env.compute_weighted_loss(ans[:, -1], train_data, update=True, weight=current_weights) + mutex_loss_fn(raw_ans[:, -1])
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            del tape
                # loss = train_sum_rate(out, train_label) + 0.01 *mutex_loss_fn(out)
            # optimizer.minimize(loss, ans)
            # l1 = tf.reduce_mean(tf.reduce_sum(env.get_weighted_rates(), axis=2))
            # lh = tf.reduce_mean(tf.reduce_sum(env.get_weighted_rates(), axis=2), axis=1)[0]
            loss_hard = env.compute_weighted_loss(Harden_scheduling_user_constrained(N_rf, K, M)(ans[:,-1]), train_data, weight=current_weights, update=False)
            train_loss(weight_sr)
            train_hard_loss(loss_hard)
            print("soft result: ", train_loss.result())
            print("hard result: ", train_hard_loss.result())
            print("on-off loss: ", on_off_loss)
        if i%check == 0:
            input_mod = valid_data
            scheduled_output = []
            raw_output = []
            for batches in range(0, math.floor(valid_data.shape[0]/10)):
                scheduled_output_temp, raw_output_temp = model(input_mod[batches * 10:(batches + 1) * 10, :, :])
                scheduled_output.append(scheduled_output_temp)
                raw_output.append(raw_output_temp)
            scheduled_output = tf.concat(scheduled_output, axis = 0)
            raw_output = tf.concat(raw_output, axis = 0)
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
                    optimizer = tf.keras.optimizers.Adam(lr=lr)
                elif improvement <= 0.0001 and lr < 0.001 and i > pre_train:
                    break
        else:
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), 0])
    np_data.save()

def grid_search_with_mutex_loss_weighted_sumrate_train_multitask(K, N_rf = 8):
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
                   "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum":Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum,
                   "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights":Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights}
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/Apr5th/K{}/multitasklearning/Nrf={}_equalWeight_and_power_law_weight".format(K, N_rf)
    positional_config = "trained_models/Apr5th/K{}/user_positions.npy".format(K)
    fname_template = fname_template_template + "{}"
    # problem Definition
    pre_train = 0
    M = 64
    # K = 100
    seed = 100
    # N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0
    ############################### generate data ###############################
    valid_data = gen_realistic_data(positional_config, 1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    ################################ hyperparameters ###############################
    EPOCHS = 100000
    lr = 0.001
    N = 30 # number of
    rounds = 2
    sample_size = 10
    temp = 0.1
    check = 200
    # episodes = 50
    alpha = .05
    model = FDD_agent_more_G_with_weights(M, K, 5, N_rf, True, max_val)
    # model = FDD_agent_more_G(M, K, 5, N_rf, True, max_val)
    # model = tf.keras.models.load_model("trained_models/Feb8th/user_loc0/on_user_loc_0_Nrf={}.h5".format(N_rf), custom_objects=custome_obj)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    # env = Weighted_sumrate_model(K, M, N_rf, N, alpha, hard_decision=False, loss_fn=Sum_rate_utility_WeiCui_seperate_user_stable(K, M, 1))
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
    env = Weighted_sumrate_model(K, M, N_rf, N, alpha, hard_decision=False)
    for i in range(0, EPOCHS):
        train_hard_loss.reset_states()
        # generate training data
        train_data = gen_realistic_data(positional_config, N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        ###################### testing with validation set ######################
        current_weights = get_realistic_weight_distribution(train_data)
        for e in range(0, rounds):
            with tf.GradientTape(persistent=True) as tape:
                temp = 0.5 * np.exp(-4.5 / rounds * e) * tf.maximum(0.0, ((200.0-i)/200.0)) + 0.1
                temp = np.float32(temp)
                train_hard_loss.reset_states()
                train_loss.reset_states()
                ###################### running model for random weight ######################
                input_mod = tf.concat([train_data, tf.complex(tf.expand_dims(current_weights, axis=2), 0.0)], axis = 2)
                # input_mod = train_data * tf.complex(tf.expand_dims(current_weights, axis=2), 0.0)
                ans, raw_ans = model(input_mod) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                out_raw = tf.transpose(raw_ans, [0, 1, 3, 2])
                out_raw = tf.reshape(out_raw[:,-1], [N * N_rf, K*M])
                sm = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw)
                out_raw = sm.sample(sample_size)
                out_raw = tf.reshape(out_raw, [sample_size, N, N_rf, K*M])
                out = tf.reduce_sum(out_raw, axis=2)
                out = tf.reshape(out, [sample_size*N, K*M])
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [sample_size,1, 1, 1]), [sample_size*N, K, M])
                weight = tf.reshape(tf.tile(tf.expand_dims(current_weights, axis=0), [sample_size,1, 1]), [sample_size*N, K])
                weight_sr = env.compute_weighted_loss(out, train_label, weight=weight, update=False)
                ###################### running model for constant weight ######################
                input_mod_sr_only = tf.concat([train_data, tf.complex(
                    tf.expand_dims(tf.ones(current_weights.shape, dtype=tf.float32), axis=2), 0.0)], axis=2)
                ans2, raw_ans2 = model(input_mod_sr_only)  # raw_ans is in the shape of (N, passes, M*K, N_rf)
                out_raw2 = tf.transpose(raw_ans2, [0, 1, 3, 2])
                out_raw2 = tf.reshape(out_raw2[:, -1], [N * N_rf, K * M])
                sm2 = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw2)
                out_raw2 = sm2.sample(sample_size)
                out_raw2 = tf.reshape(out_raw2, [sample_size, N, N_rf, K * M])
                out2 = tf.reduce_sum(out_raw2, axis=2)
                out2 = tf.reshape(out2, [sample_size * N, K * M])
                weight2 = tf.ones([sample_size * N, K], dtype=tf.float32)
                weight_sr2 = env.compute_weighted_loss(out2, train_label, weight=weight2, update=False)
                loss = weight_sr + weight_sr2 + mutex_loss_fn(raw_ans[:, -1]) + mutex_loss_fn(raw_ans2[:, -1])
                # loss = env.compute_weighted_loss(ans[:, -1], train_data, update=True, weight=current_weights) + mutex_loss_fn(raw_ans[:, -1])
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            del tape
                # loss = train_sum_rate(out, train_label) + 0.01 *mutex_loss_fn(out)
            # optimizer.minimize(loss, ans)
            # l1 = tf.reduce_mean(tf.reduce_sum(env.get_weighted_rates(), axis=2))
            # lh = tf.reduce_mean(tf.reduce_sum(env.get_weighted_rates(), axis=2), axis=1)[0]
            loss_hard = env.compute_weighted_loss(Harden_scheduling_user_constrained(N_rf, K, M)(ans[:,-1]), train_data, weight=current_weights, update=False)
            train_loss(weight_sr)
            train_hard_loss(loss_hard)
            print("sum rate: ", train_loss.result())
            print("weighted sum rate: ", tf.reduce_mean(weight_sr2))
            print("hard result: ", train_hard_loss.result())
        if i%check == 0:
            current_weights = get_realistic_weight_distribution(valid_data)
            input_mod = tf.concat([valid_data, tf.complex(tf.expand_dims(current_weights, axis=2), 0.0)], axis=2)
            scheduled_output = []
            raw_output = []
            for batches in range(0, math.floor(valid_data.shape[0]/10)):
                batch_input = input_mod[batches * 10:(batches + 1) * 10, :, :]
                scheduled_output_temp, raw_output_temp = model()
                scheduled_output.append(scheduled_output_temp)
                raw_output.append(raw_output_temp)
            scheduled_output = tf.concat(scheduled_output, axis = 0)
            raw_output = tf.concat(raw_output, axis = 0)
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
                    optimizer = tf.keras.optimizers.Adam(lr=lr)
                elif improvement <= 0.0001 and lr < 0.001 and i > pre_train:
                    break
        else:
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), 0])
    np_data.save()
def grid_search_with_mutex_loss_weighted_sumrate_train_random_K1s_modify_gain(K, N_rf = 8, Kon=8):
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
                   "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum":Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum,
                   "Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights":Per_link_Input_modification_most_G_raw_self_more_interference_mean2sum_with_weights_different_weights}
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/Apr5th/K{}/trained_with_0_1_weights_8_on/random_binary_NRF={}_biggerbatch_Kon={}".format(K, N_rf, Kon)
    positional_config = "trained_models/Apr5th/K{}/user_positions.npy".format(K)
    fname_template = fname_template_template + "{}"
    # problem Definition
    pre_train = 0
    M = 64
    # K = 100
    seed = 100
    # N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0
    ############################### generate data ###############################
    valid_data = gen_realistic_data(positional_config, 1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    ################################ hyperparameters ###############################
    EPOCHS = 100000
    lr = 0.001
    N = 60 # number of
    rounds = 8
    sample_size = 10
    temp = 0.1
    check = 200
    # episodes = 50
    alpha = .05
    # model = FDD_agent_more_G_with_weights(M, K, 5, N_rf, True, max_val)
    model = FDD_agent_more_G(M, K, 5, N_rf, True, max_val)
    # model = tf.keras.models.load_model("trained_models/Feb8th/user_loc0/on_user_loc_0_Nrf={}.h5".format(N_rf), custom_objects=custome_obj)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    # env = Weighted_sumrate_model(K, M, N_rf, N, alpha, hard_decision=False, loss_fn=Sum_rate_utility_WeiCui_seperate_user_stable(K, M, 1))
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
    env = Weighted_sumrate_model(K, M, N_rf, N, alpha, hard_decision=False)
    for i in range(0, EPOCHS):
        train_hard_loss.reset_states()
        # generate training data
        train_data = gen_realistic_data(positional_config, N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        ###################### testing with validation set ######################
        weight_indices = []
        for h in range(0, N):
           weight_indices.append(np.random.choice(K, (1, max(N_rf, Kon)), replace=False))
        weight_indices = np.concatenate(weight_indices, axis=0)
        current_weights = tf.one_hot(weight_indices, K)
        current_weights = tf.reduce_sum(current_weights, axis=1)
        for e in range(0, rounds):
            with tf.GradientTape(persistent=True) as tape:
                temp = 0.5 * np.exp(-4.5 / rounds * e) * tf.maximum(0.0, ((200.0-i)/200.0)) + 0.1
                temp = np.float32(temp)
                train_hard_loss.reset_states()
                train_loss.reset_states()
                ###################### model post-processing ######################
                # input_mod = tf.concat([train_data, tf.complex(tf.expand_dims(current_weights, axis=2), 0.0)], axis = 2)
                input_mod = train_data * tf.complex(tf.expand_dims(current_weights, axis=2), 0.0)
                ans, raw_ans = model(input_mod) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                out_raw = tf.transpose(raw_ans, [0, 1, 3, 2])
                out_raw = tf.reshape(out_raw[:,-1], [N * N_rf, K*M])
                sm = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw)
                out_raw = sm.sample(sample_size)
                out_raw = tf.reshape(out_raw, [sample_size, N, N_rf, K*M])
                out = tf.reduce_sum(out_raw, axis=2)
                out = tf.reshape(out, [sample_size*N, K*M])
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [sample_size,1, 1, 1]), [sample_size*N, K, M])
                weight = tf.reshape(tf.tile(tf.expand_dims(current_weights, axis=0), [sample_size,1, 1]), [sample_size*N, K])
                ###################### model post-processing ######################
                weight_sr = env.compute_weighted_loss(out, train_label, weight=weight, update=False)
                tiled_weight = tf.reshape(tf.tile(tf.expand_dims(weight, axis=2), (1, 1, M)), (sample_size*N, K*M))
                on_off_loss = tf.reduce_sum(tf.square(tiled_weight - out) * (1.0 - tiled_weight), axis=1)
                on_off_loss = tf.reduce_mean(on_off_loss)
                loss = weight_sr + mutex_loss_fn(raw_ans[:, -1]) + on_off_loss
                # loss = env.compute_weighted_loss(ans[:, -1], train_data, update=True, weight=current_weights) + mutex_loss_fn(raw_ans[:, -1])
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            del tape
                # loss = train_sum_rate(out, train_label) + 0.01 *mutex_loss_fn(out)
            # optimizer.minimize(loss, ans)
            # l1 = tf.reduce_mean(tf.reduce_sum(env.get_weighted_rates(), axis=2))
            # lh = tf.reduce_mean(tf.reduce_sum(env.get_weighted_rates(), axis=2), axis=1)[0]
            loss_hard = env.compute_weighted_loss(Harden_scheduling_user_constrained(N_rf, K, M)(ans[:,-1]), train_data, weight=current_weights, update=False)
            train_loss(weight_sr)
            train_hard_loss(loss_hard)
            print("soft result: ", train_loss.result())
            print("hard result: ", train_hard_loss.result())
            print("on-off loss: ", on_off_loss)
        if i%check == 0:
            input_mod = valid_data
            scheduled_output = []
            raw_output = []
            for batches in range(0, math.floor(valid_data.shape[0]/10)):
                scheduled_output_temp, raw_output_temp = model(input_mod[batches * 10:(batches + 1) * 10, :, :])
                scheduled_output.append(scheduled_output_temp)
                raw_output.append(raw_output_temp)
            scheduled_output = tf.concat(scheduled_output, axis = 0)
            raw_output = tf.concat(raw_output, axis = 0)
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
                    optimizer = tf.keras.optimizers.Adam(lr=lr)
                elif improvement <= 0.0001 and lr < 0.001 and i > pre_train:
                    break
        else:
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), 0])
    np_data.save()
def grid_search_with_mutex_loss_episodic_new_archi_fast_train(N_rf = 8):
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
    fname_template_template = "trained_models/Feb8th/user_loc0/weighted_sumrate/prior_weight_then_with_weight={}".format(N_rf)
    fname_template = fname_template_template + "{}"
    # problem Definition
    pre_train = 600
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
    rounds = 8
    sample_size = 20
    temp = 0.1
    check = 100
    episodes = 50
    alpha = .3
    model = FDD_agent_more_G_with_weights(M, K, 5, N_rf, True, max_val)
    # model = tf.keras.models.load_model("trained_models/Feb8th/user_loc0/on_user_loc_0_Nrf={}.h5".format(N_rf), custom_objects=custome_obj)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    env = Weighted_sumrate_model(K, M, N_rf, N, alpha, hard_decision=False)
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
        if i >= pre_train:
            ###################### training happens here ######################
            check = 30
            for e in range(0, rounds):
                env.reset()
                for episode in range(episodes):
                    with tf.GradientTape(persistent=True) as tape:
                        temp = 0.5 * np.exp(-4.5 / rounds * e) * tf.maximum(0.0, ((200.0-i)/200.0)) + 0.1
                        temp = np.float32(temp)
                        train_hard_loss.reset_states()
                        train_loss.reset_states()
                        ###################### model post-processing ######################
                        input_mod = tf.concat([train_data, tf.complex(tf.expand_dims(env.get_weight(), axis=2), 0.0)], axis = 2)
                        ans, raw_ans = model(input_mod) # raw_ans is in the shape of (N, passes, M*K, N_rf)
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
                        env.increment()
                        env.compute_weighted_loss(ans[:, -1], train_data, update=True)
                    print(tf.reduce_mean(tf.reduce_sum(env.rates, axis=2)))
                    # loss = train_sum_rate(out, train_label) + 0.01 *mutex_loss_fn(out)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
                # optimizer.minimize(loss, ans)
                l1 = tf.reduce_mean(tf.reduce_sum(env.rates, axis=2))
                lh = tf.reduce_mean(tf.reduce_sum(env.rates, axis=2), axis=1)[0]
                train_loss(l1)
                train_hard_loss(lh)
                print("\n===============overall=================\n",
                      l1,lh)
        else:
            train_hard_loss.reset_states()
            # generate training data
            ###################### training happens here ######################
            for e in range(0, rounds):
                with tf.GradientTape(persistent=True) as tape:
                    temp = 0.5 * np.exp(-4.5 / rounds * e) * tf.maximum(0.0, ((200.0 - i) / 200.0)) + 0.1
                    temp = np.float32(temp)
                    train_hard_loss.reset_states()
                    train_loss.reset_states()
                    ###################### model post-processing ######################
                    input_mod = tf.concat([train_data, tf.complex(tf.ones([N, K, 1], dtype=tf.float32), 0.0)],
                                          axis=2)
                    ans, raw_ans = model(input_mod)  # raw_ans is in the shape of (N, passes, M*K, N_rf)
                    out_raw = tf.transpose(raw_ans, [0, 1, 3, 2])
                    out_raw = tf.reshape(out_raw[:, -1], [N * N_rf, K * M])
                    sm = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw)
                    out_raw = sm.sample(sample_size)
                    out_raw = tf.reshape(out_raw, [sample_size, N, N_rf, K * M])
                    out = tf.reduce_sum(out_raw, axis=2)
                    out = tf.reshape(out, [sample_size * N, K * M])
                    train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [sample_size, 1, 1, 1]),
                                             [sample_size * N, K, M])
                    ###################### model post-processing ######################
                    # loss = train_label(out, train_label) + mutex_loss_fn(raw_ans[:, -1])
                    loss = sum_rate(out, train_label) + 0.01*mutex_loss_fn(raw_ans[:, -1])
                gradients = tape.gradient(loss, model.trainable_variables)

                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                train_loss(loss)
                train_hard_loss(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(ans[:, -1]), train_data))
                print(train_hard_loss.result(), train_loss.result())
        ###################### testing with validation set ######################
        if i%check == 0:
            input_mod=tf.concat([valid_data, tf.complex(tf.ones([valid_data.shape[0], K, 1], dtype=tf.float32), 0.0)],
                                          axis=2)
            scheduled_output, raw_output = model.predict(input_mod, batch_size=N)
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
                elif improvement <= 0.0001 and lr < 0.001 and i > pre_train:
                    break
        else:
            np_data.log(i, [train_hard_loss.result(), train_loss.result(), 0])
    np_data.save()
if __name__ == "__main__":
    M = 64
    K = 20
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
    for N_rf_to_search in [6, 4]:
        grid_search_with_mutex_loss_weighted_sumrate_train_multitask(40, N_rf_to_search)
        grid_search_with_mutex_loss_weighted_sumrate_train_multitask(20, N_rf_to_search)
        # grid_search_with_mutex_loss_weighted_sumrate_train_random_0_1_modify_gain(60, N_rf_to_search)
        # grid_search_with_mutex_loss_weighted_sumrate_train_random_0_1_modify_gain(40, N_rf_to_search)
