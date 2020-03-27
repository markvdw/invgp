import numpy as np
import tensorflow as tf

import gpflow


def load_mnist(digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    mnist = tf.keras.datasets.mnist.load_data()
    full_X_train, full_X_test = [x[0].reshape(-1, 28 * 28).astype(gpflow.config.default_float()) / 255.0
                                 for x in mnist]
    full_Y_train, full_Y_test = [x[1].reshape(-1, 1).astype(gpflow.config.default_int())
                                 for x in mnist]
    select_train = np.isin(full_Y_train[:, 0], digits)
    X, Y = [x[select_train, :] for x in [full_X_train, full_Y_train]]
    select_test = np.isin(full_Y_test[:, 0], digits)
    Xt, Yt = [x[select_test, :] for x in [full_X_test, full_Y_test]]
    return (X, Y), (Xt, Yt)
