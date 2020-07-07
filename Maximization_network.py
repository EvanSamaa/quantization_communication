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
    train_loss(loss)
    # train_throughput(labels, predictions, features)
    train_accuracy(labels, predictions)
def test_step_with_annealing(features, labels, N):
    # features_mod = tf.ones((features.shape[0], 1)) * N
    # features_mod = tf.concat((features_mod, features), axis=1)
    features_mod = tf.ones((features.shape[0], features.shape[1], 1)) * N
    features_mod = tf.concat((features_mod, features), axis=2)
    predictions = model(features_mod)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)
    # test_throughput(labels, predictions, features)
if __name__ == "__main__":
    # test_model()
    for i in range(0, 10):
        fname_template_template = "./trained_models/Jul 6th/k=2, LSTM, he_init_uniform/2_user_1_qbit_LSTM_encoder_tanh(relu)_seed={}"
        fname_template = fname_template_template.format(i) + "{}"
        N = 5000
        k = 2
        L = 1
        EPOCHS = 20000
        tf.random.set_seed(i)
        graphing_data = np.zeros((EPOCHS, 8))
        # model = tf.keras.models.load_model("trained_models/Sept 22_23/Data_gen_LSTM_10_cell.h5")
        model = F_create_LSTM_encoding_model_with_annealing(k, L, (k, 24))
        # model = F_create_encoding_model_with_annealing(k, L, (k, 24))
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # loss_object = tf.keras.losses.Hinge()
        # loss_object = ThroughputLoss()
        optimizer = tf.keras.optimizers.Adam()
        submodel = Model(inputs=model.input, outputs=model.get_layer("tf_op_layer_concat").output)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_throughput = ExpectedThroughput(name='train_throughput')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_throughput = ExpectedThroughput(name='test_throughput')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_acc")

        max_acc = -1
        test_ds = gen_channel_quality_data_float_encoded(100, k)
        current_acc = 0
        for epoch in range(EPOCHS):
            # Reset the metrics at the start of the next epoch
            # train_ds = gen_data(N, k, 0, 1, N)
            train_ds = gen_channel_quality_data_float_encoded(N, k)
            train_loss.reset_states()
            train_throughput.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()
            test_throughput.reset_states()
            for features, labels in train_ds:
                train_step(features, labels, epoch)
            for features, labels in test_ds:
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
            if train_accuracy.result() > max_acc:
                model.save(fname_template.format(".h5"))
                max_acc = train_accuracy.result()
            if train_accuracy.result() == 1 or train_accuracy.result() >=0.95:
                break
            if epoch%500 == 0:
                if epoch >= 1000:
                    improvement = graphing_data[epoch-1000: epoch-500, 0].mean() - graphing_data[epoch-500: epoch, 0].mean()
                    print("the accuracy improvement in the past 500 epochs is ", improvement)
                    if improvement <= 0.001:
                        break
        # fname_template = "~/quantization_communication/trained_models/Sept 25th/Data_gen_encoder_L10_hard_tanh{}"
        np.save(fname_template.format(".npy"), graphing_data)
        tf.keras.backend.clear_session()
        print("Training end")