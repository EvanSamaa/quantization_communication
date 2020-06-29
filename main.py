import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Softmax, Input, ThresholdedReLU, Flatten
from tensorflow.keras.activations import sigmoid
from util import *
from BinaryEncodingModel import *
from models import *

def freeze_encoder_layers(model):
    for layer in model.layers:
        if layer.name[0:7] == "encoder":
            layer.trainable = False
        elif layer.name[0:7] == "decoder":
            layer.trainable = True
    return model
def freeze_decoder_layers(model):
    for layer in model.layers:
        if layer.name[0:7] == "decoder":
            layer.trainable = False
        elif layer.name[0:7] == "encoder":
            layer.trainable = True
    return model
def unfreeze_all(model):
    for layer in model.layers:
        layer.trainable = True
    return model
def train_step(features, labels, N=None):
    if N != None:
        return train_step_with_annealing(features, labels, N)
    with tf.GradientTape() as tape:
        predictions = model(features)
        # quantization = submodel(features)
        # predictions = model(ranking_transform(features))
        loss = loss_object(labels, predictions)
        # loss_2 = encoding_loss(quantization)
        loss = loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    # train_throughput(labels, predictions, features)
    train_accuracy(labels, predictions)
def test_step(features, labels, N=None):
    if N != None:
        return test_step_with_annealing(features, labels, N)
    predictions = model(features)
    # predictions = model(ranking_transform(features))
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)
    # test_throughput(labels, predictions, features)
def train_step_with_annealing(features, labels, N):
    features_mod = tf.ones((features.shape[0], 1)) * N
    features_mod = tf.concat((features_mod, features), axis=1)
    with tf.GradientTape() as tape:
        predictions = model(features_mod)
        # predictions = model(ranking_transform(features))
        # quantization = submodel(features_mod)
        loss = loss_object(labels, predictions)
        # loss_2 = encoding_loss(quantization)
        loss = loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    # train_throughput(labels, predictions, features)
    train_accuracy(labels, predictions)
def test_step_with_annealing(features, labels, N):
    features_mod = tf.ones((features.shape[0], 1)) * N
    features_mod = tf.concat((features_mod, features), axis=1)
    predictions = model(features_mod)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)
    # test_throughput(labels, predictions, features)
if __name__ == "__main__":
    # test_model()
    # A[2]
    N = 10000
    k = 2
    L = 1
    switch = 20
    EPOCHS = 20000
    tf.random.set_seed(80)
    graphing_data = np.zeros((EPOCHS, 8))
    # model = binary_encoding_model((9,), 1)
    model = LSTM_loss_function(k=1, input_shape=[10000, 3])
    # submodel = Model(inputs=model.input, outputs=model.get_layer("tf_op_layer_Sign").output)
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_object = tf.keras.losses.MeanSquaredError()
    # encoding_loss = Encoding_distance()
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
    train_accuracy = Regression_Accuracy(name="train_acc")
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    # test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_acc")
    test_accuracy = Regression_Accuracy(name="test_accuracy")
    # train_ds = gen_data(N, k, 0, 1, N).shuffle(buffer_size=1000)
    train_ds = gen_encoding_data(N=1000)
    test_ds = gen_encoding_data(N=100)
    current_acc = 0
    for epoch in range(EPOCHS):
        # if epoch % switch == 0 and epoch % (2*switch) == 0:
        #     print("training decoder")
        #     model = unfreeze_all(model)
        # elif epoch % switch == 0 and epoch % (2*switch) != 0:
        #     print("training encoder")
        #     model = freeze_decoder_layers(model)
        # Reset the metrics at the start of the next epoch
        # train_ds = gen_number_data()
        train_loss.reset_states()
        # train_throughput.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        # test_throughput.reset_states()
        for features, labels in train_ds:
            train_step(features, labels)
            # train_step(features, labels, epoch)
        for features, labels in test_ds:
            test_step(features, labels)
            # test_step(features, labels, epoch)
        template = 'Epoch {}, Loss: {}, Accuracy:{}, max: {}, expected:{}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result(),
                              0,
                              0,
                              test_loss.result(),
                              test_accuracy.result()))
        graphing_data[epoch, 0] = train_loss.result()
        graphing_data[epoch, 1] = train_accuracy.result()
        # graphing_data[epoch, 2] = train_throughput.result()[0]
        # graphing_data[epoch, 3] = train_throughput.result()[1]
        graphing_data[epoch, 4] = test_loss.result()
        graphing_data[epoch, 5] = test_accuracy.result()
        # graphing_data[epoch, 6] = test_throughput.result()[0]
        # graphing_data[epoch, 7] = test_throughput.result()[1]
        if epoch%100 == 0:
            if epoch >= 200:
                improvement = graphing_data[epoch-100: epoch, 1].mean() - graphing_data[epoch-200: epoch-100, 1].mean()
                print("the accuracy improvement in the past 500 epochs is ", improvement)
                if improvement <= 0.0001:
                    break
    fname_template = "./trained_models/Sept 29/LSTM_Loss_function{}"
    # fname_template = "~/quantization_communication/trained_models/Sept 25th/Data_gen_encoder_L10_hard_tanh{}"
    np.save(fname_template.format(".npy"), graphing_data)
    model = unfreeze_all(model)
    # model = replace_tanh_with_sign(model, binary_encoding_model_regularization, k=8)
    model.save(fname_template.format(".h5"))
    print("Training data and ")