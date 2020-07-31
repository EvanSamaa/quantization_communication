import tensorflow as tf
from util import *
from models import *
import numpy as np
import scipy as sp
# from matplotlib import pyplot as plt
def train_step(features, labels, N=None):
    if N != None:
        return train_step_with_annealing(features, labels, N)
    with tf.GradientTape() as tape:
        # f_features = float_to_floatbits(features, complex=True)
        # predictions = model(f_features)
        predictions = model(features)
        # loss_1 = loss_object_1(predictions, features, display=np.random.choice([False, False], p=[0.1, 0.9]))
        loss_1 = loss_object_1(predictions, features)
        loss_2 = loss_object_2(predictions, features)
        loss_3 = loss_object_3(predictions)
        loss = loss_1 + 1*loss_2 + loss_3
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss_1)
    train_binarization_loss(loss_3)
    train_VS(loss_2)
def test_step(features, labels, N=None):
    if N != None:
        return test_step_with_annealing(features, labels, N)
    # f_features = float_to_floatbits(features, complex=True)
    # predictions = model(f_features)
    predictions = model(features)
    # predictions = Masking_with_learned_weights_soft(K, M, sigma2_n, N_rf)(predictions)
    t_loss_1 = loss_object_1(predictions, features)
    t_loss_2 = loss_object_1(predictions, features)
    test_binarization_loss(t_loss_2)
def train_step_with_annealing(features, labels, N):
    # features_mod = tf.ones((features.shape[0], 1)) * N
    # features_mod = tf.concat((features_mod, features), axis=1)
    features_mod = tf.ones((features.shape[0], features.shape[1], 1)) * N
    features_mod = tf.concat((features_mod, features), axis=2)
    with tf.GradientTape() as tape:
        predictions = model(features_mod)
        # predictions = model(ranking_transform(features))
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # optimizer.apply_gradients(gradients, model.trainable_variables)
    train_loss(loss)
    # train_throughput(labels, predictions, features)
    train_accuracy(labels, predictions)
def test_step_with_annealing(features, labels, N):
    # features_mod = tf.ones((features.shape[0], 1)) * N
    # features_mod = tf.concat((features_mod, features), axis=1)
    features_mod = tf.ones((features.shape[0], features.shape[1], 1)) * N
    features_mod = tf.concat((features_mod, features), axis=2)
    predictions = model(features_mod)
    # print(predictions)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)
    # test_throughput(labels, predictions, features)

def random_complex(shape, sigma2):
    A_R = np.random.normal(0, sigma2, shape)
    A_R = np.array(A_R, dtype=complex)
    A_R.imag = np.random.normal(0, sigma2, shape)
    return A_R
if __name__ == "__main__":
    fname_template = "trained_models/Jul 30th/sumrate_VS_ranked_state_change_5_times_noise=0.1_magnitude_input{}"
    check = 400
    # problem Definition
    N = 1000
    M = 40
    K = 10
    B = 10
    seed = 200
    N_rf = 5
    sigma2_h = 6.3
    sigma2_n = 0.1
    # hyperparameters
    EPOCHS = 100000
    tf.random.set_seed(seed)
    np.random.seed(seed)
    # loss_object_1 = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    loss_object_1 = Sum_rate_utility_RANKING(K, M, sigma2_n, N_rf)
    # loss_object_2 = Sum_rate_utility_WeiCui_wrong_axis(K, M, sigma2_n)
    loss_object_2 = Verti_sum_utility_RANKING(K, M, sigma2_n, N_rf)
    loss_object_3 = Binarization_regularization(K, N, M, N_rf)
    # model = FDD_softmax_k_times_common_dnn(M, K, N_rf)
    # model = FDD_softmax_k_times_hard_output_with_magnitude(M, K, N_rf)
    # model = FDD_softmax_k_times_with_magnitude(M, K, N_rf)
    # model = FDD_ranked_softmax_common_DNN(M, K, N_rf)
    # model = FDD_ranked_LSTM_softmax(M, K, N_rf)
    model = FDD_ranked_softmax_state_change(M, K, N_rf)
    # model = Floatbits_FDD_model_softmax(M, K, B)
    # model = FDD_softmax_with_unconstraint_soft_masks(M, K, B, k=N_rf)
    optimizer = tf.keras.optimizers.Adam(0.0001)
    # for data visualization
    graphing_data = np.zeros((EPOCHS, 4))
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_binarization_loss = tf.keras.metrics.Mean(name='train_loss')
    train_VS = tf.keras.metrics.Mean(name='test_loss')
    test_binarization_loss = tf.keras.metrics.Mean(name='train_loss')
    # begin setting up training loop
    max_acc = 10000
    test_ds = generate_link_channel_data(100, K, M)
    # training Loop
    for epoch in range(EPOCHS):
        # train_features = generate_link_channel_data(500, K, M)
        train_features = generate_link_channel_data(500, K, M)
        # data recording features
        train_loss.reset_states()
        train_binarization_loss.reset_states()
        train_VS.reset_states()
        test_binarization_loss.reset_states()
        train_step(features=train_features, labels=None)
        template = 'Epoch {}, Loss: {}, binarization_lost:{}, VS Loss: {}, Test binarization_lost: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_binarization_loss.result(),
                              train_VS.result(),
                              test_binarization_loss.result()))
        graphing_data[epoch, 0] = train_loss.result()
        graphing_data[epoch, 1] = train_binarization_loss.result()
        graphing_data[epoch, 2] = train_VS.result()
        graphing_data[epoch, 3] = test_binarization_loss.result()
        if train_loss.result() < max_acc:
            model.save(fname_template.format(".h5"))
            max_acc = train_loss.result()
        if epoch % check == 0:
            if epoch >= (check*2):
                improvement = graphing_data[epoch - (check*2): epoch - check, 0].mean() - graphing_data[epoch - check: epoch, 0].mean()
                print("the accuracy improvement in the past 500 epochs is ", improvement)
                if improvement <= 0.001:
                    break
    np.save(fname_template.format(".npy"), graphing_data)
    tf.keras.backend.clear_session()
    print("Training end")

    