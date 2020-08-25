import tensorflow as tf
from util import *
from models import *
import numpy as np
import scipy as sp
from keras_adabound.optimizers import AdaBound


# from matplotlib import pyplot as plt
def train_step(features, labels, N=None, epoch=0):
    if N == 0:
        with tf.GradientTape() as tape:
            predictions = model(features)
            print(tf.argmax(predictions[0]), tf.reduce_max(predictions[0]))
            # predictions = Masking_with_learned_weights_soft(K, M, sigma2_n, k=N_rf)(predictions)
            loss_3 = tf.reduce_sum(predictions, axis=1) - N_rf
            loss = supervised_loss(predictions, labels) + tf.square(loss_3)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss_object_1(predictions, features))
        # train_binarization_loss(loss_3)
        return
    elif N == 1:
        with tf.GradientTape() as tape:
            predictions = model(features)
            # predictions = Masking_with_learned_weights_soft(K, M, sigma2_n, k=N_rf)(predictions)
            loss = supervised_loss(predictions, Harden_scheduling(k=N_rf)(predictions))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss_object_1(predictions, features))
        # train_binarization_loss(loss_3)
        train_VS(loss_object_2(predictions, features))
        train_hard_loss(loss_object_1(Harden_scheduling(k=N_rf)(predictions), features))
        return
    with tf.GradientTape() as tape:
        scheduled_output, z_qq, z_e, reconstructed_input = model(features)
        # scheduled_output, z_q_b, z_e_b, z_q_t, z_e_t, reconstructed_input = model(features)
        # reconstructed_input, z_q_b, z_e_b, z_q_t, z_e_t = model(features)
        # predictions_hard = predictions + tf.stop_gradient(Harden_scheduling(k=N_rf)(predictions) - predictions)
        # mask = tf.stop_gradient(Harden_scheduling(k=N_rf)(scheduled_output))
        loss_1 = 0
        # loss_3 = tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(features))
        # loss_1 = tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(features))
        loss_2 = vae_loss.call(z_qq, z_e)
        # loss_4 = tf.keras.losses.CategoricalCrossentropy()(scheduled_output, mask)
        for i in range(0, scheduled_output.shape[1]):
            # predictions = predictions + tf.stop_gradient(Harden_scheduling(k=N_rf)(predictions) - predictions)
            sr = sum_rate(scheduled_output[:, i], features)
            # mask = tf.stop_gradient(Harden_scheduling(k=N_rf)(scheduled_output[:, i]))
            # ce = tf.keras.losses.CategoricalCrossentropy()(scheduled_output[:, i], mask)
            loss_1 = loss_1 + tf.exp(tf.constant(-scheduled_output.shape[1]+1+i, dtype=tf.float32)) * sr
            # loss_4 = loss_4 + tf.exp(tf.constant(-predictions.shape[1]+1+i, dtype=tf.float32)) * ce
            # loss_2 = loss_2 + tf.exp(tf.constant(-predictions.shape[1]+1+i, dtype=tf.float32)) * vs
        print("==============================")
        loss = loss_1 + loss_2
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(sum_rate(scheduled_output[:, -1], features))
    # train_loss(loss_1)
    # train_binarization_loss(loss_4)
    train_hard_loss(sum_rate(Harden_scheduling(k=N_rf)(scheduled_output[:, -1]), features))
if __name__ == "__main__":
    fname_template = "trained_models/Aug24th/test{}"
    check = 500
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2
    # problem Definition
    N = 50
    M = 40
    K = 10
    B = 10
    E = 10
    B_t = 10
    E_t = 30
    seed = 100
    N_rf = 3
    sigma2_h = 6.3
    sigma2_n = 0.1
    # hyperparameters
    EPOCHS = 100000
    tf.random.set_seed(seed)
    np.random.seed(seed)
    # model = CSI_reconstruction_model_seperate_decoders(M, K, B, E, N_rf, 6, more=1, qbit=0)
    # model = CSI_reconstruction_VQVAE2(M, K, B, E, N_rf, 6, B_t=B_t, E_t=E_t, more=1)
    # model = Feedbakk_FDD_model_scheduler_VAE2(M, K, B, E, N_rf, 6, B_t=B_t, E_t=E_t, more=1, output_all=True)
    model = Feedbakk_FDD_model_scheduler(M, K, B, E, N_rf, 6, more=1, qbit=0, output_all=True)
    # model = CSI_reconstruction_model(M, K, B, E, N_rf, 6)

    vae_loss = VAE_loss_general(False)
    sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    # optimizer = tf.keras.optimizers.SGD(lr=0.001)
    # for data visualization
    graphing_data = np.zeros((EPOCHS, 4))
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_binarization_loss = tf.keras.metrics.Mean(name='train_loss')
    train_VS = tf.keras.metrics.Mean(name='test_loss')
    train_hard_loss = tf.keras.metrics.Mean(name='train_loss')
    # begin setting up training loop
    max_acc = 10000
    # training Loop
    for epoch in range(EPOCHS):
        # ======== ======== data recording features ======== ========
        train_loss.reset_states()
        train_binarization_loss.reset_states()
        train_VS.reset_states()
        train_hard_loss.reset_states()
        # ======== ======== training step ======== ========
        train_features = generate_link_channel_data(N, K, M)
        train_step(train_features, None, training_mode, epoch=epoch)
        # train_step(features=train_features, labels=None)
        template = 'Epoch {}, Loss: {}, binarization_lost:{}, VS Loss: {}, Hard Loss: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_binarization_loss.result(),
                              train_VS.result(),
                              train_hard_loss.result()))
        graphing_data[epoch, 0] = train_loss.result()
        graphing_data[epoch, 1] = train_binarization_loss.result()
        graphing_data[epoch, 2] = train_VS.result()
        graphing_data[epoch, 3] = train_hard_loss.result()
        if train_loss.result() < max_acc:
            max_acc = train_loss.result()
            model.save(fname_template.format(".h5"))
        if epoch % check == 0:
            if epoch >= (SUPERVISE_TIME) and epoch >= (check * 2):
                improvement = graphing_data[epoch - (check * 2): epoch - check, 0].mean() - graphing_data[
                                                                                            epoch - check: epoch,
                                                                                            0].mean()
                print("the accuracy improvement in the past 500 epochs is ", improvement)
                if improvement <= 0.001:
                    break
    np.save(fname_template.format(".npy"), graphing_data)
    tf.keras.backend.clear_session()
    print("Training end")


