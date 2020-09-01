import tensorflow as tf
from util import *
from models import *
import numpy as np
import scipy as sp
from keras_adabound.optimizers import AdaBound

custome_obj = {'Closest_embedding_layer': Closest_embedding_layer, 'Interference_Input_modification': Interference_Input_modification,
                   'Interference_Input_modification_no_loop': Interference_Input_modification_no_loop,
                   "Interference_Input_modification_per_user":Interference_Input_modification_per_user,
                   "Closest_embedding_layer_moving_avg":Closest_embedding_layer_moving_avg}
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
    with tf.GradientTape(persistent=True) as tape:
        # scheduled_output, z_qq, z_e, reconstructed_input = model(features)
        # reconstructed_input, z_qq, z_e= model(features)
        # scheduled_output, z_q_b, z_e_b, z_q_t, z_e_t, reconstructed_input = model(features)
        # reconstructed_input, z_q_b, z_e_b, z_q_t, z_e_t = model(features)
        # scheduled_output, per_user_softmaxes, overall_softmax = model(features)
        # scheduled_output, z_qq, z_e, reconstructed_input, per_user_softmaxes, overall_softmax = model(features)
        scheduled_output, raw_output, z_qq, z_e, reconstructed_input = model(features)
        # predictions_hard = predictions + tf.stop_gradient(Harden_scheduling(k=N_rf)(predictions) - predictions)
        # mask = tf.stop_gradient(Harden_scheduling(k=N_rf)(overall_softmax))
        # loss_1 = tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(features))
        loss_1 = 0
        loss_3 = tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(features))
        loss_2 = vae_loss.call(z_qq, z_e)
        loss_4 = 0
        for i in range(0, scheduled_output.shape[1]):
            sr = sum_rate(scheduled_output[:, i], features)
            loss_1 = loss_1 + tf.exp(tf.constant(-scheduled_output.shape[1]+1+i, dtype=tf.float32)) * sr
            # loss_1 = loss_1 + sr
            # ce = All_softmaxes_CE(N_rf)(per_user_softmaxes[:, i], overall_softmax[:, i])
            ce = All_softmaxes_CE_general(N_rf, K, M)(raw_output[:, i])
            loss_4 = loss_4 + tf.exp(tf.constant(-scheduled_output.shape[1]+1+i, dtype=tf.float32)) * ce
            mask = tf.stop_gradient(Harden_scheduling(k=N_rf)(scheduled_output[:, i]))
            ce = tf.keras.losses.CategoricalCrossentropy()(scheduled_output[:, i], mask)
            loss_4 = loss_4 + tf.exp(tf.constant(-scheduled_output.shape[1]+1+i, dtype=tf.float32)) * ce
            # loss_2 = loss_2 + tf.exp(tf.constant(-predictions.shape[1]+1+i, dtype=tf.float32)) * vs
        # # print("==============================")
        loss = loss_1 + loss_2 + loss_3
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    gradients2 = tape.gradient(loss_4, model.get_layer("model_2").trainable_variables)
    optimizer2.apply_gradients(zip(gradients2, model.get_layer("model_2").trainable_variables))
    train_loss(sum_rate(scheduled_output[:, -1], features))
    # train_loss(loss_3)
    train_binarization_loss(loss_3)
    train_hard_loss(sum_rate(Harden_scheduling(k=N_rf)(scheduled_output[:, -1]), features))
    del tape
if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    fname_template = "trained_models/Aug31/M=64_K=50/N_rf=3/VAEB=1x128E=4+1x512_per_linkx6_alt+seperate_doubleCE_loss+MP{}"
    check = 500
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2
    # problem Definition
    N = 5
    M = 64
    K = 50
    B = 1
    E = 4
    B_t = 10
    E_t = 30
    seed = 100
    N_rf = 3
    sigma2_h = 6.3
    sigma2_n = 0.1
    # hyperparameters
    EPOCHS = 100000

    for i in range(0, 1):
        train_VS = tf.keras.metrics.Mean(name='test_loss')
        tf.random.set_seed(seed)
        np.random.seed(seed)
        # model = CSI_reconstruction_model_seperate_decoders(M, K, B, E, N_rf, 6, more=3, qbit=0)
        # model = CSI_reconstruction_VQVAE2(M, K, B, E, N_rf, 6, B_t=B_t, E_t=E_t, more=1)
        # model = Feedbakk_FDD_model_scheduler_VAE2(M, K, B, E, N_rf, 6, B_t=B_t, E_t=E_t, more=1, output_all=True)
        # model = Feedbakk_FDD_model_scheduler(M, K, B, E, N_rf, 6, more=32, qbit=0, output_all=False)
        # model = FDD_per_user_architecture_return_all_softmaxes(M, K, 6, N_rf)
        # model = Feedbakk_FDD_model_scheduler_per_user(M, K, B, E, N_rf, 3, more=32, qbit=0, output_all=True)
        # model = tf.keras.models.load_model("trained_models/Aug27th/B4x8E10code_stacking+input_mod.h5", custom_objects=custome_obj)
        # model = CSI_reconstruction_model(M, K, B, E, N_rf, 6, more=32)
        # model = Feedbakk_FDD_model_scheduler_per_user(M, K, B, E, N_rf, 6, 32, output_all=True)
        # model = FDD_per_link_archetecture_more_granular(M, K, 6, N_rf, output_all=True)
        model = Feedbakk_FDD_model_scheduler(M, K, B, E, N_rf, 6, more=128, qbit=0, output_all=True)
        print(model.summary())
        vae_loss = VAE_loss_general(False)
        sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
        optimizer = tf.keras.optimizers.Adam(lr=0.0001)
        optimizer2 = tf.keras.optimizers.Adam(lr=0.0001)
        # optimizer = tf.keras.optimizers.SGD(lr=0.001)
        # for data visualization
        graphing_data = np.zeros((EPOCHS, 4))
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_binarization_loss = tf.keras.metrics.Mean(name='train_loss')
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
            if train_hard_loss.result() < max_acc:
                max_acc = train_hard_loss.result()
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


