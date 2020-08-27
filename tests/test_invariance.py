import numpy as np
import pytest
import tensorflow as tf

import gpflow
import invgp
from invgp.kernels.orbits import SwitchXY


@pytest.mark.parametrize("inducing_variable", [gpflow.inducing_variables.InducingPoints,
                                               invgp.inducing_variables.ConvolvedInducingPoints])
def test_invariant_model(inducing_variable):
    X = np.array([[1.0, 0.0], [1.0, 0.03], [1.0, 1.0]])
    Y = np.array([[1.0], [0.9], [0.0]])

    kernel = invgp.kernels.Invariant(gpflow.kernels.SquaredExponential(), SwitchXY())
    Z = kernel.orbit(X).numpy().reshape(-1, 2)
    m = gpflow.models.SGPR((X, Y), kernel, inducing_variable=inducing_variable(Z))

    opt = gpflow.optimizers.Scipy()
    opt.minimize(tf.function(lambda: -m.elbo()),
                 m.trainable_variables,
                 options=dict(maxiter=1000))

    Xt = np.random.randn(100, 2)
    Xt_orbit = tf.reshape(m.kernel.orbit(Xt), (-1, Xt.shape[1]))
    preds = m.predict_f(Xt_orbit)[0].numpy().reshape(len(Xt), m.kernel.orbit.orbit_size)
    assert np.all(preds[:, 0] == preds[:, 1])


@pytest.mark.parametrize("orbit", [invgp.kernels.orbits.ImageRot90(),
                                   invgp.kernels.orbits.ImageRotQuant(90, interpolation_method="BILINEAR")])
def test_rot_kernels(orbit):
    # This is not that great a test for ImageRotQuant...
    mnist = tf.keras.datasets.mnist.load_data()[0]
    mnist_X_full = mnist[0].reshape(-1, 28 * 28).astype(gpflow.config.default_float()) / 255.0
    mnist_Y_full = mnist[1].reshape(-1, 1).astype(gpflow.config.default_int())

    rotq_k = invgp.kernels.Invariant(gpflow.kernels.SquaredExponential(), orbit)
    m = gpflow.models.GPR((mnist_X_full[:2, :], mnist_Y_full[:2, :]), rotq_k)

    img_orbit = orbit(mnist_X_full[None, 0, :])[0, :, :]
    pm, pv = m.predict_f(img_orbit)
    assert np.allclose(pm, pm[0, 0])
    assert np.allclose(pv, pv[0, 0])
