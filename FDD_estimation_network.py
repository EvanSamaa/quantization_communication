import tensorflow as tf
from util import *
from models import *
import numpy as np
import scipy as sp
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
        # f_features = float_to_floatbits(features, complex=True)
        # predictions = model(f_features)
        predictions = model(features)
        print(tf.argmax(predictions[0]), tf.reduce_max(predictions[0]))
        # predictions = predictions + tf.stop_gradient(binary_activation(predictions) - predictions)
        # predictions = predictions + tf.stop_gradient(Harden_scheduling(k=N_rf)(predictions) - predictions)
        # predictions = Masking_with_learned_weights_soft(K, M, sigma2_n, k=N_rf)(predictions)
        # loss_1 = loss_object_1(predictions, features, display=np.random.choice([False, False], p=[0.1, 0.9]))
        loss_1 = loss_object_1(predictions, features)
        loss_2 = loss_object_2(predictions, features)
        loss_3 = tf.reduce_mean(tf.reduce_sum(binary_activation(predictions), axis=1))
        loss = loss_1 + loss_2
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss_1)
    train_binarization_loss(loss_3)
    # train_VS(loss_3)
    # train_hard_loss(loss_object_1(Harden_scheduling(k=N_rf)(predictions), features))
    train_hard_loss(loss_object_1(binary_activation(predictions), features))

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
    fname_template = "trained_models/Aug9th/Wei_cui_like_model_SGD{}"
    check = 500
    SUPERVISE_TIME = 0
    training_mode = 2
    swap_delay = check/2
    # problem Definition
    N = 500
    M = 40
    K = 10
    B = 10
    seed = 200
    N_rf = 3
    sigma2_h = 6.3
    sigma2_n = 0.1
    # hyperparameters
    EPOCHS = 100000
    tf.random.set_seed(seed)
    np.random.seed(seed)
    supervised_loss = tf.keras.losses.CategoricalCrossentropy()
    loss_object_1 = Sum_rate_utility_WeiCui(K, M, sigma2_n)
    # loss_object_1 = Sum_rate_utility_RANKING(K, M, sigma2_n, N_rf)
    loss_object_2 = Sum_rate_utility_WeiCui_wrong_axis(K, M, sigma2_n)
    # model = FDD_k_times_with_sigmoid_and_penalty(M, K, k=1)
    # model = FDD_distributed_then_general_architecture(M, K, k=3, N_rf=N_rf)
    # model = FDD_per_link_archetecture(M, K, k=3, N_rf=N_rf)
    # model = FDD_Dumb_model(M, K, k=1, N_rf=N_rf)
    model = FDD_per_link_archetecture_sigmoid(M, K, k=3, N_rf=N_rf)
    # optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    optimizer = tf.keras.optimizers.SGD(lr=0.0005)
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
        if epoch <= SUPERVISE_TIME:
            train_ds = generate_supervised_link_channel_data(N, K, M, N_rf)
            for features, labels in train_ds:
                train_step(features, labels, 0)
        else:
            train_features = generate_link_channel_data(N, K, M)
            train_step(train_features, None, training_mode, epoch = epoch)
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
            model.save(fname_template.format(".h5"))
            max_acc = train_loss.result()
        if epoch % check == 0:
            if epoch >= (SUPERVISE_TIME) and epoch >= (check*2):
                improvement = graphing_data[epoch - (check*2): epoch - check, 0].mean() - graphing_data[epoch - check: epoch, 0].mean()
                print("the accuracy improvement in the past 500 epochs is ", improvement)
                if improvement <= 0.01:
                    break
    np.save(fname_template.format(".npy"), graphing_data)
    tf.keras.backend.clear_session()
    print("Training end")

    