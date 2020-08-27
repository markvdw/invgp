import gpflow
from gpflow.utilities import set_trainable, multiple_assign, read_values
import numpy as np
import numpy.random as rnd
import tensorflow as tf
from gpflow.models import SVGP

from invgp.models.SampleSVGP import SampleSVGP
from invgp_experiments import utils

np.random.seed(0)

import numpy as np


def test_regression():
    # generate data
    X = np.vstack((np.random.uniform(0, 6, 50)[:, None], 8))
    Y = np.sin(2 * X) + 0.1 * np.cos(7 * X) + np.random.randn(*X.shape) * 0.4
    train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    train_dataset = train_dataset.shuffle(1024).batch(len(X))

    # initialize SVGP models
    SVGP_model, sample_SVGP_model, matheron_sample_SVGP_model = [
        m(
            gpflow.kernels.SquaredExponential(),
            gpflow.likelihoods.Gaussian(),
            inducing_variable=X.copy(),
            num_data=len(X),
            **kw,
        )
        for m, kw in [(SVGP, {}), (SampleSVGP, {"matheron_sampler": False}), (SampleSVGP, {"matheron_sampler": True})]
    ]

    # train SVGP model
    set_trainable(SVGP_model.inducing_variable, False)
    train_iter = iter(train_dataset.repeat())
    training_loss = SVGP_model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.keras.optimizers.Adam(0.01)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, SVGP_model.trainable_variables)

    elbo_hist = []
    for step in range(500):
        optimization_step()
        minibatch_elbo = -training_loss().numpy()
        print("Step: %s, Mini batch elbo: %s" % (step, minibatch_elbo))
        elbo_hist.append(minibatch_elbo)

    # Commented out code is for tuning the test. For the test, we want a relatively large variance for f, **at the
    # training data locations**. This is how we get a large variance for the ELBO, which is the strongest test for
    # correctness. To verify this, we plot the 1D model. We also plot the ELBOs over time during training, so we can
    # choose an appropriate training time.
    # import matplotlib.pyplot as plt
    # utils.plot_1d_model(plt.gca(), SVGP_model, data=(X, Y))
    # plt.show()
    # plt.plot(elbo_hist)
    # plt.show()

    # initialize sample SVGP models with fitted parameters from "regular" SVGP
    trained_params = read_values(SVGP_model)
    multiple_assign(sample_SVGP_model, trained_params)
    multiple_assign(matheron_sample_SVGP_model, trained_params)

    # compare elbos
    SVGP_model_elbo = SVGP_model.elbo((X, Y)).numpy()
    assert SVGP_model.elbo((X, Y)).numpy() == SVGP_model_elbo  # Ensure that there is no stochasticity in reference

    sample_SVGP_model_elbos = [sample_SVGP_model.elbo((X, Y)).numpy() for _ in range(100)]
    np.testing.assert_allclose(SVGP_model_elbo, np.mean(sample_SVGP_model_elbos), rtol=0.01, atol=0.0)
    assert np.std(sample_SVGP_model_elbos) > 0

    matheron_sample_SVGP_model_elbos = [matheron_sample_SVGP_model.elbo((X, Y)).numpy() for _ in range(100)]
    np.testing.assert_allclose(SVGP_model_elbo, np.mean(matheron_sample_SVGP_model_elbos), rtol=0.01, atol=0.0)
    assert np.std(matheron_sample_SVGP_model_elbos) > 0
