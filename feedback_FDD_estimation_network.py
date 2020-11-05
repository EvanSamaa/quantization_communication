import tensorflow as tf
from util import *
from models import *
import numpy as np
import scipy as sp
from keras_adabound.optimizers import AdaBound
# from matplotlib import pyplot as plt
custome_obj = {'Closest_embedding_layer': Closest_embedding_layer,
               'Interference_Input_modification': Interference_Input_modification,
               'Interference_Input_modification_no_loop': Interference_Input_modification_no_loop,
               "Interference_Input_modification_per_user": Interference_Input_modification_per_user,
               "Closest_embedding_layer_moving_avg": Closest_embedding_layer_moving_avg,
               "Per_link_Input_modification_more_G": Per_link_Input_modification_more_G,
               "Per_link_Input_modification_more_G_less_X": Per_link_Input_modification_more_G_less_X,
               "Per_link_Input_modification_even_more_G": Per_link_Input_modification_even_more_G,
               "Per_link_Input_modification_compress_XG": Per_link_Input_modification_compress_XG,
               "Per_link_Input_modification_compress_XG_alt": Per_link_Input_modification_compress_XG_alt,
               "Per_link_Input_modification_more_G_alt_2": Per_link_Input_modification_more_G_alt_2,
               "Per_link_Input_modification_compress_XG_alt_2": Per_link_Input_modification_compress_XG_alt_2,
               "Per_link_Input_modification_most_G": Per_link_Input_modification_most_G,
               "Per_link_sequential_modification": Per_link_sequential_modification,
               "Per_link_sequential_modification_compressedX": Per_link_sequential_modification_compressedX,
               "Per_link_Input_modification_most_G_raw_self": Per_link_Input_modification_most_G_raw_self,
               "Reduced_output_input_mod":Reduced_output_input_mod,
               "TopPrecoderPerUserInputMod":TopPrecoderPerUserInputMod,
               "X_extends": X_extends}
# from matplotlib import pyplot as plt
def train_step(features, labels, N=None, epoch=0, lr_boost=1.0):
    with tf.GradientTape(persistent=True) as tape:
        # compressed_G, position_matrix = G_compress(features, 2)
        # scheduled_output, raw_output = model([features, compressed_G, position_matrix])
        scheduled_output, raw_output = model(features)
        # scheduled_output = model(features)
        # mask = tf.stop_gradient(Harden_scheduling(k=N_rf)(overall_softmax))
        # scheduled_output, raw_output, z_qq, z_e, reconstructed_input = model(features)
        # loss_1 = tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(features))

        # input_mod = tf.abs(features)
        # norm = tf.reduce_max(tf.keras.layers.Reshape((K * M,))(input_mod), axis=1, keepdims=True)
        # input_mod = tf.divide(input_mod, tf.expand_dims(norm, axis=1))
        loss_1 = 0
        # loss_3 = 10*tf.keras.losses.MeanSquaredError()(reconstructed_input, tf.abs(features))
        # loss_2 = 30*vae_loss.call(z_qq, z_e)
        # mask = tf.stop_gradient(Harden_scheduling_user_constrained(N_rf, K, M, default_val=0)(scheduled_output))
        factor = {1:1.0, 2:1.0, 3:1.0, 4:0.5, 5:0.5, 6:0.25, 7:0.25, 8:0.25}
        # loss_4 = tf.reduce_mean(tf.square(tf.multiply(scheduled_output, 1.0-scheduled_output)), axis=1)
        loss_4 = 0
        for i in range(0, scheduled_output.shape[1]):
            # sr = sum_rate_train(scheduled_output[:, i], features)
            # loss_1 = loss_1 + tf.exp(tf.constant(-scheduled_output.shape[1]+1+i, dtype=tf.float32)) * sr * lr_boost
            # ce = All_softmaxes_MSE_general(N_rf, K, M)(raw_output[:, i])
            # ce = All_softmaxes_CE_general(N_rf, K, M)(raw_output[:, i])
            # mask = partial_feedback_pure_greedy_model(N_rf, 32, 10, M, K, sigma2_n)(features)
            # ce = tf.keras.losses.CategoricalCrossentropy()(scheduled_output[:, i]/N_rf, mask/N_rf)
            # loss_4 = loss_4 + tf.exp(tf.constant(-scheduled_output.shape[1]+1+i, dtype=tf.float32)) * ce

            if i < scheduled_output.shape[1]-1:
                mask = tf.stop_gradient(Harden_scheduling_user_constrained(1, K, M, default_val=0)(scheduled_output[:, i]))
                sr = sum_rate_hard(scheduled_output[:, i], mask, features)
                sr2 = sum_rate_train(scheduled_output[:, i], features)
                sr_i = sum_rate_interference(scheduled_output[:, i], features)
                ce = tf.reduce_mean(tf.square(tf.multiply(scheduled_output[:, i], 1.0-scheduled_output[:, i])), axis=1)
                loss_1 = loss_1 + 0.01 * sr_i
                # loss_4 = loss_4 + factor[N_rf]*tf.exp(tf.constant(-scheduled_output.shape[1]+1+i, dtype=tf.float32)) * ce * lr_boost
                # ce_lambda = tf.reduce_mean(lambda_var_1 * (tf.multiply(scheduled_output[:, i], 1.0-scheduled_output[:, i])), axis=1)
                # reshaped_X = tf.keras.layers.Reshape((K, M))(scheduled_output[:, i])
                # user_constraint = tf.minimum(tf.square(tf.reduce_sum(reshaped_X, axis=1) - 1), tf.square(tf.reduce_sum(reshaped_X, axis=1)))
                # user_constraint = tf.reduce_mean(user_constraint, axis=1)
                # user_constraint_lambda = tf.reduce_mean(user_constraint_lambda, axis=1)
                # loss_4 = loss_4 + ce
            else:
                loss_1 = loss_1 + sum_rate_train(scheduled_output[:, i], features)
        # # print("==============================")
        # mask = tf.stop_gradient(Harden_scheduling_user_constrained(1, K, M, default_val=0)(scheduled_output))
        # loss_4 += tf.keras.losses.CategoricalCrossentropy()(scheduled_output/N_rf, mask/N_rf)
        loss = loss_1
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # gradients_2 = tape.gradient(loss_4, model.get_layer("model_1").trainable_variables)
    # optimizer.apply_gradients(zip(gradients_2, model.get_layer("model_1").trainable_variables))
    train_loss(sum_rate(scheduled_output[:, -1], features))
    # train_loss(loss_3)
    # train_binarization_loss(loss_3)
    train_hard_loss(sum_rate(Harden_scheduling_user_constrained(N_rf, K, M)(scheduled_output[:, -1]), features))
    del tape
    return train_hard_loss.result()
if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    # fname_template = "trained_models/Sept23rd/Nrf=4/Nrf={}normaliza_input_0p25CE+residual_more_G{}"
    fname_template = "trained_models/OCT30/Nrf=4/seeding={}moreG_model_onehot_iter+raw_self+minimal_x+min_interference{}"
    check = 250
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check / 2
    # problem Definition
    N = 50
    M = 64
    K = 50
    B = 1
    E = 4
    more = 32
    seed = 100
    N_rf = 4
    sigma2_h = 6.3
    sigma2_n = 1.0
    # hyperparameters
    EPOCHS = 100000
    # EPOCHS = 1
    mores = [1]
    Es = [1]
    for j in Es:
        for i in mores:
            train_VS = tf.keras.metrics.Mean(name='test_loss')
            tf.random.set_seed(i)
            np.random.seed(i)
            N_rf = 4
            # model = CSI_reconstruction_model_seperate_decoders(M, K, B, E, N_rf, 6, more=3, qbit=0)
            # model = CSI_reconstruction_VQVAE2(M, K, B, E, N_rf, 6, B_t=B_t, E_t=E_t, more=1)
            # model = Feedbakk_FDD_model_scheduler_VAE2(M, K, B, E, N_rf, 6, B_t=B_t, E_t=E_t, more=1, output_all=True)
            # model = Feedbakk_FDD_model_scheduler(M, K, B, E, N_rf, 6, more=more, qbit=0, output_all=False)
            # model = FDD_per_user_architecture_return_all_softmaxes(M, K, 6, N_rf)
            # model = Feedbakk_FDD_model_scheduler_per_user(M, K, B, E, N_rf, 3, more=32, qbit=0, output_all=True)
            # model = tf.keras.models.load_model("trained_models/Aug27th/B4x8E10code_stacking+input_mod.h5", custom_objects=custome_obj)
            # model = CSI_reconstruction_model(M, K, B, E, N_rf, 6, more=32)
            # model = Feedbakk_FDD_model_scheduler_per_user(M, K, B, E, N_rf, 6, 32, output_all=True)
            # model = FDD_per_link_archetecture_more_granular(M, K, 6, N_rf, output_all=True)
            # model =  FDD_per_link_archetecture_more_G_distillation(M, K, 6, N_rf, output_all=True)
            # model = FDD_per_link_2Fold(M, K, 6, N_rf, output_all=True)
            model = FDD_per_link_archetecture_more_G(M, K, 6, N_rf, output_all=True)
            lambda_var_1 = tf.Variable(1.0, trainable=True)
            lambda_var_2 = tf.Variable(1.0, trainable=True)
            lambda_var_3 = tf.Variable(1.0, trainable=True)

            # model = FDD_RNN_model(M, K, N_rf)

            # model = FDD_per_link_2Fold(M, K, 6, N_rf, output_all=True)
            # model = Top2Precoder_model(M, K, 4, N_rf, 2)
            # model = CSI_reconstruction_model_seperate_decoders_input_mod(M, K, 6, N_rf, output_all=True, more=more)
            # model = FDD_reduced_output_space(M, K, N_rf)
            # model = FDD_distributed_then_general_architecture(M, K, k=2, N_rf=N_rf, output_all=False)
            # model = Feedbakk_FDD_mcodel_scheduler(M, K, B, E, N_rf, 6, more=more, qbit=0, output_all=True)
            # model = Feedbakk_FDD_model_scheduler_naive(M, K, B, E, N_rf, 6, more=more, qbit=0, output_all=True)
            vae_loss = VAE_loss_general(False)
            sum_rate = Sum_rate_utility_WeiCui(K, M, sigma2_n)
            sum_rate_hard = Sum_rate_utility_hard(K, M, sigma2_n)
            sum_rate_train = Sum_rate_utility_WeiCui(K, M, sigma2_n)
            sum_rate_interference = Sum_rate_interference(K, M, sigma2_n)
            optimizer = tf.keras.optimizers.Adam(lr=0.01)
            optimizer2 = tf.keras.optimizers.Adam(lr=0.001)
            # optimizer = tf.keras.optimizers.SGD(lr=0.001)
            # for data visualization
            graphing_data = np.zeros((EPOCHS, 4))
            train_loss = tf.keras.metrics.Mean(name='train_loss')
            train_binarization_loss = tf.keras.metrics.Mean(name='train_loss')
            train_hard_loss = tf.keras.metrics.Mean(name='train_loss')
            valid_sum_rate = tf.keras.metrics.Mean(name='valid_loss')
            # begin setting up training loop
            max_acc = 10000
            max_acc_loss = 10000
            # training Loop
            valid_data = generate_link_channel_data(1000, K, M, Nrf=N_rf)
            for epoch in range(EPOCHS):
                # ======== ======== data recording features ======== ========
                train_loss.reset_states()
                train_binarization_loss.reset_states()
                train_VS.reset_states()
                train_hard_loss.reset_states()
                valid_sum_rate.reset_states()
                # ======== ======== training step ======== ========
                # if epoch % 20 == 0:
                train_features = generate_link_channel_data(N, K, M, N_rf)
                current_result = train_step(train_features, None, training_mode, epoch=epoch)
                # out = partial_feedback_pure_greedy_model(N_rf, 32, 2, M, K, sigma2_n)(train_features)
                # if current_result >= graphing_data[max(epoch - check, 0):max(0, epoch-1), 3].mean():
                # if True:
                #     for m in range(0, 10000):
                #         train_hard_loss.reset_states()
                #         current_result = train_step(train_features, None, training_mode, epoch=epoch, lr_boost=1)
                #         print(train_loss.result(), current_result)
                train_step(features=train_features, labels=None)
                # A[2]
                template = 'Epoch {}, Loss: {}, binarization_lost:{}, VS Loss: {}, Hard Loss: {}'
                print(template.format(epoch,
                                      train_loss.result(),
                                      train_binarization_loss.result(),
                                      train_VS.result(),
                                      train_hard_loss.result()))
                graphing_data[epoch, 0] = train_loss.result()
                graphing_data[epoch, 1] = train_binarization_loss.result()
                # graphing_data[epoch, 2] = train_VS.result()
                graphing_data[epoch, 3] = train_hard_loss.result()
                # if train_hard_loss.result() < max_acc_loss:
                #     max_acc_loss = train_hard_loss.result()
                #     model.save(fname_template.format(i, "_max_train2.h5"))
                #     tim = tf.keras.models.load_model(fname_template.format(i, "_max_train2.h5"), custom_objects=custome_obj)

                if epoch % check == 0:
                    # compressed_G, position_matrix = G_compress(valid_data, 2)
                    # scheduled_output, raw_output = model.predict_on_batch([valid_data, compressed_G, position_matrix])
                    scheduled_output, raw_output = model.predict(valid_data, batch_size=N)
                    out = sum_rate(Harden_scheduling_user_constrained(N_rf, K, M, default_val=0)(scheduled_output[:, -1]), tf.abs(valid_data))
                    valid_sum_rate(out)
                    graphing_data[epoch, 2] = valid_sum_rate.result()
                    if valid_sum_rate.result() < max_acc:
                        max_acc = valid_sum_rate.result()
                        model.save(fname_template.format(i, ".h5"))
                    if epoch >= (SUPERVISE_TIME) and epoch >= (check * 2):
                        improvement = graphing_data[epoch + 1 - (check * 2): epoch - check + 1, 2].min() - graphing_data[
                                                                                                    epoch - check + 1: epoch + 1,
                                                                                                    2].min()
                        counter = 0
                        for asldk in graphing_data[0:epoch+1, 2]:
                            if asldk != 0:
                                print(counter, asldk)
                            counter = counter + 1
                        print("the improvement in the past 500 epochs is: ", improvement)
                        print("the validation SR is: ", valid_sum_rate.result())
                        if improvement <= 0.0001:
                            break
            np.save(fname_template.format(i, ".npy"), graphing_data)
            tf.keras.backend.clear_session()
            print("Training end")


