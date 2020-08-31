import gpflow
import numpy as np
import numpy.random as rnd
import pytest
import tensorflow as tf
from gpflow.models import SVGP
from gpflow.utilities import read_values, multiple_assign, set_trainable

import invgp
from invgp.inducing_variables.invariant_convolution_domain import StochasticConvolvedInducingPoints
from invgp.models.SampleSVGP import SampleSVGP

np.random.seed(0)


@pytest.mark.parametrize(
    "inducing_variables_creator", [lambda X: StochasticConvolvedInducingPoints(X.copy()), lambda X: X.copy()],
)
def test_invariant_regression(inducing_variables_creator):
    # generate datapoints
    X = np.random.uniform(-3, 3, 400)[:, None]
    X = np.reshape(X, [200, 2])  # 2-dimensional input
    Y = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)[..., None] + np.random.randn(len(X), 1) * 0.1
    train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    train_dataset = train_dataset.shuffle(1024).batch(len(X))

    # initialize SVGP model
    # inducing_variables_creator = lambda: StochasticConvolvedInducingPoints(X.copy())
    # inducing_variables_creator = lambda: X.copy()
    kernel_creator = lambda: invgp.kernels.StochasticInvariant(
        basekern=gpflow.kernels.SquaredExponential(), orbit=invgp.kernels.orbits.SwitchXY()
    )

    SVGP_model, sample_SVGP_model, matheron_sample_SVGP_model = [
        m(
            kernel_creator(),
            gpflow.likelihoods.Gaussian(),
            inducing_variable=inducing_variables_creator(X),
            num_data=len(X),
            **kw,
        )
        for m, kw in [(SVGP, {}), (SampleSVGP, {"matheron_sampler": False}), (SampleSVGP, {"matheron_sampler": True})]
    ]

    # train SVGP model
    train_iter = iter(train_dataset.repeat())
    training_loss = SVGP_model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.keras.optimizers.Adam(0.01)
    set_trainable(SVGP_model.inducing_variable, False)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, SVGP_model.trainable_variables)

    elbo_hist = []
    for step in range(2000):
        optimization_step()
        if step % 10 == 0:
            minibatch_elbo = -training_loss().numpy()
            print("Step: %s, Mini batch elbo: %s" % (step, minibatch_elbo))
            elbo_hist.append(minibatch_elbo)

    # initialize sample SVGP model with fitted parameters from SVGP
    trained_params = read_values(SVGP_model)
    multiple_assign(sample_SVGP_model, trained_params)
    multiple_assign(matheron_sample_SVGP_model, trained_params)
    gpflow.utilities.print_summary(SVGP_model)
    gpflow.utilities.print_summary(sample_SVGP_model)
    gpflow.utilities.print_summary(matheron_sample_SVGP_model)

    # compare elbos
    SVGP_model_elbo = SVGP_model.elbo((X, Y))

    sample_SVGP_model_elbos = [sample_SVGP_model.elbo((X, Y)).numpy() for _ in range(10)]
    expected_sample_elbo = np.mean(sample_SVGP_model_elbos)
    np.testing.assert_allclose(
        SVGP_model_elbo, expected_sample_elbo, rtol=0.05, atol=0.0
    )  # the tolerance is picked somewhat randomly

    if type(SVGP_model.inducing_variable) is gpflow.inducing_variables.InducingPoints:
        with pytest.raises(NotImplementedError):
            matheron_sample_SVGP_model.elbo((X, Y))
    else:
        matheron_sample_SVGP_model_elbos = [matheron_sample_SVGP_model.elbo((X, Y)).numpy() for _ in range(10)]
        expected_matheron_sample_elbo = np.mean(matheron_sample_SVGP_model_elbos)
        np.testing.assert_allclose(
            SVGP_model_elbo, expected_matheron_sample_elbo, rtol=0.05, atol=0.0
        )  # the tolerance is picked somewhat randomly
