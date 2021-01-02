import tensorflow as tf
from tf_agents.distributions import gumbel_softmax
from util import *
from models import *
import numpy as np
import scipy as sp
from keras_adabound.optimizers import AdaBound
def grid_search(bits = 8):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template_template = "trained_models/better_quantizer/STE{}bits"
    fname_template = fname_template_template.format(bits) + "{}"
    check = 250
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2

    # problem Definition
    M = 64
    K = 50
    B = 4
    E = 30
    more = bits
    seed = 100
    N_rf = 8
    sigma2_h = 6.3
    sigma2_n = 1.0
    res = 6
    ############################### generate data ###############################
    valid_data = generate_link_channel_data(1000, K, M, Nrf=N_rf)
    garbage, max_val = Input_normalization_per_user(tf.abs(valid_data))
    q_valid_data = tf.abs(valid_data) / max_val
    q_valid_data = tf.where(q_valid_data > 1.0, 1.0, q_valid_data)
    q_valid_data = tf.round(q_valid_data * (2 ** res - 1)) / (2 ** res - 1) * max_val
    ################################ hyperparameters ###############################
    EPOCHS = 1000000
    lr = 0.001
    N = 400 # number of
    rounds = 1000
    sample_size = 100
    temp = 0.1
    check = 1000
    model = CSI_reconstruction_model_seperate_decoders_naive(M, K, B, E, N_rf, more=more, avg_max=max_val)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    ################################ Metrics  ###############################
    sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    train_reconstruction_loss = tf.keras.metrics.Mean(name='train_loss')
    ################################ storing train data in npy file  ##############################
    # the three would be first train_loss, Hardloss, and the validation loss, every 50 iterations
    max_acc = 0
    np_data = ModelTrainer(save_dir=fname_template.format(".npy"), data_cols=2, epoch=EPOCHS)
    # training loop
    for i in range(0, EPOCHS):
        # generate training data
        train_reconstruction_loss.reset_states()
        train_data = generate_link_channel_data(N, K, M, Nrf=N_rf)
        ###################### training happens here ######################
        for e in range(0, 1):
            train_reconstruction_loss.reset_states()
            with tf.GradientTape(persistent=True) as tape:
                ###################### model post-processing ######################
                # q_train_data = tf.abs(train_data)/max_val
                # q_train_data = tf.where(q_train_data > 1.0, 1.0, q_train_data)
                # q_train_data = tf.round(q_train_data * (2 ** res - 1)) / (2 ** res - 1) * max_val
                reconstructed_input = model(train_data) # raw_ans is in the shape of (N, passes, M*K, N_rf)
                train_label = tf.reshape(tf.tile(tf.expand_dims(train_data, axis=0), [100,1, 1, 1]), [100*N, K, M])
                ###################### model post-processing ######################
                loss = tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(train_data))

            gradient = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))
            # optimizer.minimize(loss, ans)
            train_reconstruction_loss(loss)
            print(train_reconstruction_loss.result())
            del tape
        ###################### testing with validation set ######################
        if i%check == 0:
            reconstructed_input = model(valid_data)
            valid_loss = tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(valid_data))
            np_data.log(i, [train_reconstruction_loss.result(), valid_loss])
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
                if improvement <= 0.0001:
                    break
        else:
            np_data.log(i, [train_reconstruction_loss.result(), 0])
    np_data.save()
if __name__ == "__main__":
    for N_rf_to_search in [16,32,64,128]:
        grid_search(N_rf_to_search)
