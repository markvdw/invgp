import numpy as np
import pytest
import tensorflow as tf

import gpflow
import invgp


@pytest.mark.parametrize("inducing_variable", [gpflow.inducing_variables.InducingPoints,
                                               invgp.inducing_variables.ConvolvedInducingPoints])
def test_invariant_model(inducing_variable):
    X = np.array([[1.0, 0.0], [1.0, 0.03], [1.0, 1.0]])
    Y = np.array([[1.0], [0.9], [0.0]])

    kernel = invgp.kernels.SwitchDimsInvariant(gpflow.kernels.SquaredExponential())
    Z = kernel.orbit(X).numpy().reshape(-1, 2)
    m = gpflow.models.SGPR((X, Y), kernel, inducing_variable=inducing_variable(Z))

    opt = gpflow.optimizers.Scipy()
    opt.minimize(tf.function(lambda: -m.log_marginal_likelihood()),
                 m.trainable_variables,
                 options=dict(maxiter=1000))

    Xt = np.random.randn(100, 2)
    Xt_orbit = tf.reshape(m.kernel.orbit(Xt), (-1, Xt.shape[1]))
    preds = m.predict_f(Xt_orbit)[0].numpy().reshape(len(Xt), m.kernel.orbit_size)
    assert np.all(preds[:, 0] == preds[:, 1])
