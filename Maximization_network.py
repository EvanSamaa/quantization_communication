import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Softmax, Input, ThresholdedReLU, Flatten
from tensorflow.keras.activations import sigmoid
import random
from util import *
from models import *
def train_step(features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        # predictions = model(ranking_transform(features))
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_throughput(labels, predictions, features)
    train_accuracy(labels, predictions)
def test_step(features, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(features)
    # predictions = model(ranking_transform(features))
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)
    test_throughput(labels, predictions, features)

if __name__ == "__main__":
    # test_model()
    # A[2]
    N = 10000
    k = 10
    EPOCHS = 100
    tf.random.set_seed(80)
    graphing_data = np.zeros((EPOCHS, 8))
    # es = tf.keras.callbacks.EarlyStopping(monitor="train_loss", mode="min", patience="30")
    # model = create_MLP_model_with_transform((k,k), k)
    # model = tf.keras.models.load_model("trained_models/N_10000_5_Layer_MLP_regression.h5")
    # model = create_LSTM_model(k, [k, 1])
    model = create_encoding_model(k, 4, (k,))
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss_object = tf.keras.losses.Hinge()
    # loss_object = Throughput()
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_throughput = ExpectedThroughput(name='train_throughput')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_throughput = ExpectedThroughput(name='test_throughput')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_acc")
    train_ds = gen_data(N, k, 0, 1).shuffle(buffer_size=1000)
    test_ds = gen_data(100, k, 0, 1)
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_throughput.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        test_throughput.reset_states()
        for features, labels in train_ds:
            train_step(features, labels)
        for features, labels in test_ds:
            test_step(features, labels)
        template = 'Epoch {}, Loss: {}, max: {}, expected:{}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
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
    fname_template = "./trained_models/N_{}_auto_encoding_MLP{}"
    np.save(fname_template.format(N, ".npy"), graphing_data)
    model.save(fname_template.format(N, ".h5"))