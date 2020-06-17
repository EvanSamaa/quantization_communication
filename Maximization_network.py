import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Softmax, Input, ThresholdedReLU, Flatten
from tensorflow.keras.activations import sigmoid
import random
def train_step(features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        # predictions = model(ranking_transform(features))
        loss = loss_object(labels, predictions, features)
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
    t_loss = loss_object(labels, predictions, features)
    test_loss(t_loss)
    test_accuracy(labels, predictions)
    test_throughput(labels, predictions, features)
def gen_data(N, k, low=0, high=1, batchsize=30):
    channel_data = tf.random.uniform((N,k), low, high)
    channel_label = tf.math.argmax(channel_data, axis=1)
    dataset = Dataset.from_tensor_slices((channel_data, channel_label)).batch(batchsize)
    return dataset

class ExpectedThroughput(tf.keras.metrics.Metric):
    def __init__(self, name='expected_throughput', **kwargs):
        super(ExpectedThroughput, self).__init__(name=name, **kwargs)
        self.max_throughput = self.add_weight(name='tp', initializer='zeros')
        self.expected_throughput = self.add_weight(name='tp2', initializer='zeros')
        self.a = tf.math.log(tf.cast(2, dtype=tf.float32))
    def update_state(self, y_true, y_pred, x):
        c_max = tf.gather(x, y_true, axis=1, batch_dims=1)
        i_max = tf.math.log(1 - 1.5*tf.math.log(1 - c_max))/self.a
        weighted_c = tf.math.multiply(Softmax()(y_pred), x)
        i_expected = tf.math.reduce_sum(tf.math.log(1 - 1.5*tf.math.log(1 - weighted_c))/self.a, axis=1)
        self.max_throughput.assign_add(tf.math.reduce_sum(i_max, axis=0))
        self.expected_throughput.assign_add(tf.math.reduce_sum(i_expected, axis=0))
    def result(self):
        return self.max_throughput, self.expected_throughput
def ThroughoutLoss():
    def through_put_loss(y_true, y_pred, x=None):
        a = tf.math.log(tf.cast(2, dtype=tf.float32))
        c_max = tf.gather(x, y_true, axis=1, batch_dims=1)
        i_max = tf.math.log(1 - 1.5 * tf.math.log(1 - c_max)) / a
        weighted_c = tf.math.multiply(Softmax()(y_pred), x)
        i_expected = tf.math.reduce_sum(tf.math.log(1 - 1.5 * tf.math.log(1 - weighted_c)) / a, axis=1)
        return tf.square(tf.math.reduce_sum(i_max, axis=0) - tf.math.reduce_sum(i_expected, axis=0))
    return through_put_loss
def Mix_loss():
    def mixloss(y_true, y_pred, x=None):
        a = tf.math.log(tf.cast(2, dtype=tf.float32))
        c_max = tf.gather(x, y_true, axis=1, batch_dims=1)
        i_max = tf.math.log(1 - 1.5 * tf.math.log(1 - c_max)) / a
        weighted_c = tf.math.multiply(Softmax()(y_pred), x)
        i_expected = tf.math.reduce_sum(tf.math.log(1 - 1.5 * tf.math.log(1 - weighted_c)) / a, axis=1)
        loss1 = tf.square(tf.math.reduce_sum(i_max, axis=0) - tf.math.reduce_sum(i_expected, axis=0))
        loss2 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred)
        return loss2 + loss1/100
    return mixloss
def create_MLP_model(input_shape, k):
    inputs = Input(shape=input_shape)
    x = perception_model(inputs, k, 5)
    model = Model(inputs, x, name="max_nn")
    return model
def create_MLP_model_with_transform(input_shape, k):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = perception_model(x, k, 5)
    model = Model(inputs, x, name="max_nn")
    return model
def create_uniformed_quantization_model(input_shape, k):
    inputs = Input(shape=input_shape)
    x = tf.round(inputs * 1000)/1000
    x = perception_model(x, k, 5)
    model = Model(inputs, x, name="max_nn_with_rounding")
def perception_model(x, output, layer, logit=True):
    for i in range(layer-1):
        x = Dense(50)(x)
        x = LeakyReLU()(x)
    x = Dense(output)(x)
    if logit:
        return x
    else:
        return Softmax(x)
def ranking_transform(x):
    out = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for k in range(x.shape[0]):
        for i in range(0, x.shape[1]):
            for j in range(0, x.shape[1]):
                if x[k, i] >= x[k, j]:
                    out[k, i, j] = 1
    return tf.convert_to_tensor(out, dtype=tf.float32)
def test_model():
    model = tf.keras.models.load_model("models/three_layer_MLP_1800_epoch.h5")
    test_throughput = ExpectedThroughput(name="test_throughput")
    expected_throughput = 0
    max_throughput = 0

    for i in range(0, 100):
        test_throughput.reset_states()
        test_ds = gen_data(100, 10, 0, 1)
        for features, labels in test_ds:
            prediction = model.predict(features)
            test_throughput(labels, prediction, features)
        expected_throughput = expected_throughput + test_throughput.result()[1]
        max_throughput = max_throughput + test_throughput.result()[0]
    print(expected_throughput/10000)
    print(max_throughput/10000)
    print(max_throughput/10000 - expected_throughput/10000)
if __name__ == "__main__":
    # test_model()
    # A[2]
    N = 10000
    k = 10
    EPOCHS = 500
    tf.random.set_seed(80)
    graphing_data = np.zeros((EPOCHS, 8))
    es = tf.keras.callbacks.EarlyStopping(monitor="train_loss", mode="min", patience="30")
    model = create_MLP_model((k,), k)
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss_object = tf.keras.losses.Hinge()
    loss_object = ThroughoutLoss()
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_throughput = ExpectedThroughput(name='train_throughput')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    test_throughput = ExpectedThroughput(name='test_throughput')
    train_ds = gen_data(N, k, 0, 1).shuffle(buffer_size=1000)
    test_ds = gen_data(100, k, 0, 1)
    # cycle = {0:(0,1), 1:(0,2), 2:(2,3), 3:(4,5), 4:(6,7), 5:(1,3)}
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
    np.save("mlp_training_throughput_loss.npy", graphing_data)
    model.save('models/three_layer_mlp_throughput_for_presenting.h5')