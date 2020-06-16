import tensorflow as tf

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print(x_train[:1])
    print(y_train[:1])
    # model declaration
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    predictions = model(x_train[:1]).numpy()
    out = tf.nn.softmax(predictions).numpy()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # this fits the training and stuff
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    # trained here
    model.fit(x_train, y_train, epochs=5)
    # this can be used to evaluate
    model.evaluate(x_test, y_test, verbose=2)

    # finish with the model and put probability at it
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
