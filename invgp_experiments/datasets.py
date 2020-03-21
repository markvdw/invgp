import numpy as np
import tensorflow as tf

import gpflow


def load_mnist(digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    mnist = tf.keras.datasets.mnist.load_data()[0]
    mnist_X_full = mnist[0].reshape(-1, 28 * 28).astype(gpflow.config.default_float()) / 255.0
    mnist_Y_full = mnist[1].reshape(-1, 1).astype(gpflow.config.default_int())
    select = np.isin(mnist_Y_full[:, 0], digits)
    X = mnist_X_full[select, :]
    Y = mnist_Y_full[select, :]
    return (X, Y), (None, None)  # Todo: Return testing data as well
