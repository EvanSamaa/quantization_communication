import tensorflow as tf
from tf_agents.distributions import gumbel_softmax
from util import *
from models import *
import numpy as np
import scipy as sp
from keras_adabound.optimizers import AdaBound

if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template = "trained_models/Dec_13/GNN_simple_model_test{}"
    check = 250
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2

    # problem Definition
    M = 64
    K = 50
    B = 4
    E = 30
    more = 64
    seed = 100
    N_rf = 4
    sigma2_h = 6.3
    sigma2_n = 1.0
    np.random.seed(seed)
    tf.random.set_seed(seed)
    ############################### generate data ###############################
    valid_data = generate_link_channel_data(1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    ################################ hyperparameters ###############################

    EPOCHS = 100000
    lr = 0.03
    N = 1 # number of
    rounds = 400
    sample_size = 100
    temp = 0.3
    check = 10
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    ################################ Metrics  ###############################
    sum_rate = Sum_rate_utility_WeiCui(K, M, 1)
    train_sum_rate = Sum_rate_utility_WeiCui(K, M, 1)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_hard_loss = tf.keras.metrics.Mean(name='train_loss')
    ################################ storing train data in npy file  ##############################
    # the three would be first train_loss, Hardloss, and the validation loss, every 50 iterations
    max_acc = 0
    np_data = ModelTrainer(save_dir=fname_template.format(".npy"), data_cols=3, epoch=EPOCHS)
    # training loop
    for i in range(0, EPOCHS):
        train_hard_loss.reset_states()
        # generate training data
        train_data = generate_link_channel_data(N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        model = tf.Variable(tf.random.normal((1, 1, K*M, N_rf)))
        ###################### training happens here ######################
        temp = 0.3
        for e in range(0, rounds):
            # if e >= 2*rounds/3:
            #     temp = 0.3
            train_loss.reset_states()
            train_hard_loss.reset_states()
            with tf.GradientTape(persistent=True) as tape:
                ###################### model post-processing #####################
                # raw_ans = tf.transpose(model, [0, 1, 3, 2])
                # out_raw = tf.reshape(raw_ans[:,-1], [N * N_rf, K*M])
                # sm = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw)
                # out_raw = sm.sample(sample_size)
                # out_raw = tf.reshape(out_raw, [sample_size, N, N_rf, K*M])
                # out = tf.reduce_sum(out_raw, axis=2)
                # out = tf.reshape(out, [sample_size*N, K*M])
                # train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [100,1, 1, 1]), [100*N, K, M])
                ###################### model post-processing ######################

                ###################### model post-processing #####################
                # if e >= 4 * rounds / 8:
                #     temp = 0.1
                # elif e >= 3 * rounds / 8:
                #     temp = 0.2
                # elif e >= 2 * rounds / 8:
                #     temp = 0.25
                raw_ans_user = tf.transpose(model, [0, 1, 3, 2])
                out_raw_user = tf.reshape(raw_ans_user[:,-1], [N * N_rf, K*M])
                sm_user = gumbel_softmax.GumbelSoftmax(temperature=temp, logits=out_raw_user)
                out_raw_user = tf.expand_dims(sm_user.sample(sample_size), axis=3)
                out_raw = tf.reshape(out_raw_user, [sample_size, N, N_rf, K*M])
                out = tf.reduce_sum(out_raw, axis=2)
                out = tf.reshape(out, [sample_size*N, K*M])
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [100,1, 1, 1]), [100*N, K, M])
                ###################### model post-processing ######################
                loss = train_sum_rate(out, train_label)
            gradients = tape.gradient(loss, model)
            optimizer.apply_gradients(zip([gradients], [model]))
            train_loss(loss)
            hard_decsition = Harden_scheduling_user_constrained(N_rf, K, M)(tf.reduce_sum(tf.keras.layers.Softmax(axis=2)(model), axis=3)[0])
            train_hard_loss(sum_rate(hard_decsition, train_label))
            print(train_hard_loss.result(),train_loss.result())

            del tape
        from matplotlib import pyplot as plt
        plt.plot(hard_decsition[0].numpy(), "-",label="hard")
        plt.plot(tf.reduce_sum(tf.keras.layers.Softmax(axis=2)(model), axis=3)[0,0].numpy(), label="soft")
        plt.legend()
        plt.show()
        ###################### testing with validation set ######################