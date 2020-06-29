import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Softmax, Input, ThresholdedReLU, Flatten
from tensorflow.keras.activations import sigmoid
import random
from util import *
import os
from models import *
def train_step(features, labels, N=None):
    if N != None:
        return train_step_with_annealing(features, labels, N)
    with tf.GradientTape() as tape:
        predictions = model(features)
        # predictions = model(ranking_transform(features))
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_throughput(labels, predictions, features)
    train_accuracy(labels, predictions)
def test_step(features, labels, N=None):
    if N != None:
        return test_step_with_annealing(features, labels, N)
    predictions = model(features)
    # predictions = model(ranking_transform(features))
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)
    test_throughput(labels, predictions, features)

def train_step_with_annealing(features, labels, N):
    features_mod = tf.ones((features.shape[0], 1)) * N
    features_mod = tf.concat((features_mod, features), axis=1)
    with tf.GradientTape() as tape:
        predictions = model(features_mod)
        # predictions = model(ranking_transform(features))
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_throughput(labels, predictions, features)
    train_accuracy(labels, predictions)
def test_step_with_annealing(features, labels, N):
    features_mod = tf.ones((features.shape[0], 1)) * N
    features_mod = tf.concat((features_mod, features), axis=1)
    predictions = model(features_mod)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)
    test_throughput(labels, predictions, features)

if __name__ == "__main__":
    # test_model()
    # A[2]
    N = 10000
    k = 2
    L = 1
    EPOCHS = 20000
    tf.random.set_seed(80)
    graphing_data = np.zeros((EPOCHS, 8))
    # model = create_MLP_model_with_transform((k,k), k)
    # model = tf.keras.models.load_model("trained_models/Sept 22_23/Data_gen_LSTM_10_cell.h5")
    # model = create_uniformed_quantization_model(10)
    # model = create_LSTM_model(k, [k, 1], 10)
    # model = create_BLSTM_model_with2states(k, [k, 1], state_size=30)
    # model = create_uniform_encoding_model(k, 10, (k,))
    # model = create_encoding_model(k, 10, (k, ))
    # model = create_encoding_model_with_annealing(k, L, (k+1, ))
    model = create_uniform_encoding_model_with_annealing(k, L, (k+1, ))
    # model = create_MLP_mean0_model([k, ], k)
    # model = create_MLP_model(input_shape=(k, ), k=k)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss_object = tf.keras.losses.Hinge()
    # loss_object = ThroughputLoss()
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_throughput = ExpectedThroughput(name='train_throughput')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_throughput = ExpectedThroughput(name='test_throughput')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_acc")
    # train_ds = gen_data(N, k, 0, 1, N).shuffle(buffer_size=1000)
    test_ds = gen_data(100, k, 0, 1)
    current_acc = 0
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_ds = gen_data(N, k, 0, 1, N)
        train_loss.reset_states()
        train_throughput.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        test_throughput.reset_states()
        for features, labels in train_ds:
            # train_step(features, labels)
            train_step(features, labels, epoch)
        for features, labels in test_ds:
            # test_step(features, labels)
            test_step(features, labels, epoch)
        template = 'Epoch {}, Loss: {}, Accuracy:{}, max: {}, expected:{}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result(),
                              train_throughput.result()[0],
                              train_throughput.result()[1],
                              test_loss.result(),
                              test_accuracy.result()))
        graphing_data[epoch, 0] = train_loss.result()
        graphing_data[epoch, 1] = train_accuracy.result()
        graphing_data[epoch, 2] = train_throughput.result()[0]
        graphing_data[epoch, 3] = train_throughput.result()[1]
        graphing_data[epoch, 4] = test_loss.result()
        graphing_data[epoch, 5] = test_accuracy.result()
        graphing_data[epoch, 6] = test_throughput.result()[0]
        graphing_data[epoch, 7] = test_throughput.result()[1]
        if epoch%100 == 0:
            if epoch >= 200:
                improvement = graphing_data[epoch-100: epoch, 1].mean() - graphing_data[epoch-200: epoch-100, 1].mean()
                print("the accuracy improvement in the past 500 epochs is ", improvement)
                if improvement <= 0.0001:
                    break
    fname_template = "./trained_models/Sept 25/k=2, L=2/Data_gen_common_encoder_L=1_k=2_tanh_annealing{}"
    # fname_template = "~/quantization_communication/trained_models/Sept 25th/Data_gen_encoder_L10_hard_tanh{}"
    np.save(fname_template.format(".npy"), graphing_data)
    model.save(fname_template.format(".h5"))
    print("Training data and ")