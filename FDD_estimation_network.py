import tensorflow as tf
from util import *
from models import *
import numpy as np
import scipy as sp
def train_step(features, labels, N=None):
    if N != None:
        return train_step_with_annealing(features, labels, N)
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss_1 = loss_object_1(predictions, features)
        loss_2 = loss_object_2(predictions)
        loss = loss_1 + loss_2
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss_1)
    train_binarization_loss(loss_2)
def test_step(features, labels, N=None):
    if N != None:
        return test_step_with_annealing(features, labels, N)
    predictions = model(features)
    # predictions = model(ranking_transform(features))
    t_loss_1 = loss_object_1(predictions, features)
    t_loss_2 = loss_object_2(predictions)
    test_loss(t_loss_1)
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
    fname_template = "trained_models/Jul 18th/Naive_model_with_per_user_softmax_and_ranking_prob{}"
    tf.keras.backend.set_floatx('float64')
    # problem Definition
    N = 1000
    M = 20
    K = 20
    B = 10
    N_rf = 5
    sigma2_h = 6.3
    sigma2_n = 0.000001
    # hyperparameters
    EPOCHS = 20000
    tf.random.set_seed(10)
    loss_object_1 = Sum_rate_utility_top_k_with_mask_from_ranking_prob(K, M, sigma2_n, N_rf)
    loss_object_2 = Binarization_regularization(K, N, M)
    # loss_object_2 = Output_Per_Receiver_Control(K, M)
    model = FDD_encoding_model_constraint_13_with_softmax(M, K, B)
    optimizer = tf.keras.optimizers.Adam()


    # for data visualization
    graphing_data = np.zeros((EPOCHS, 4))
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_binarization_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_binarization_loss = tf.keras.metrics.Mean(name='train_loss')
    # begin setting up training loop
    max_acc = 10000
    test_ds = generate_link_channel_data(100, K, M)
    # training Loop
    for epoch in range(EPOCHS):
        train_features = generate_link_channel_data(1000, K, M)
        # data recording features
        train_loss.reset_states()
        train_binarization_loss.reset_states()
        test_loss.reset_states()
        test_binarization_loss.reset_states()
        train_step(features=train_features, labels=None)
        test_step(features=train_features, labels=None)
        template = 'Epoch {}, Loss: {}, binarization_lost:{}, Test Loss: {}, Test binarization_lost: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_binarization_loss.result(),
                              test_loss.result(),
                              test_binarization_loss.result()))
        graphing_data[0] = train_loss.result()
        graphing_data[1] = train_binarization_loss.result()
        graphing_data[2] = test_loss.result()
        graphing_data[3] = test_binarization_loss.result()
        if train_loss.result() < max_acc:
            model.save(fname_template.format(".h5"))
            max_acc = train_loss.result()
        if epoch % 100 == 0:
            if epoch >= 200:
                improvement = graphing_data[epoch - 200: epoch - 100, 0].mean() - graphing_data[epoch - 100: epoch, 0].mean()
                print("the accuracy improvement in the past 500 epochs is ", improvement)
                if tf.abs(improvement) <= 0.001:
                    break
    np.save(fname_template.format(".npy"), graphing_data)
    tf.keras.backend.clear_session()
    print("Training end")

    