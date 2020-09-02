import gpflow
import numpy as np
import numpy.random as rnd
import pytest
import tensorflow as tf
from gpflow.models import SVGP
from gpflow.utilities import read_values, multiple_assign, set_trainable

import invgp
from deepkernelinv.kernels import DeepKernel
from deepkernelinv.inducing_variables import KernelSpaceInducingPoints, ConvolvedKernelSpaceInducingPoints, StochasticConvolvedKernelSpaceInducingPoints

from invgp.inducing_variables.invariant_convolution_domain import StochasticConvolvedInducingPoints, ConvolvedInducingPoints
from invgp.models.SampleSVGP import SampleSVGP

import matplotlib.pyplot as plt
from gpflow.config import default_float

np.random.seed(0)

# TESTS
# 1. train a sample SVGP model and confirm that a deep kernel with identity mapping yields the same results, for both invariant and non invariant kernels 
# (tested combinations are 
#---------------------------------------------------------------------------------------------------
# kernels                                                         inducing points
# --------------------------------------------------------------------------------------------------
# SquaredExponential+ deepkernel                                  X + KernelSpaceInducingPoints(cnn(X))
# Invariant(rbf) + Invariant(deepkernel)                          ConvolvedInducingPoints + ConvolvedKernelSpaceInducingPoints(cnn(X))
# StochasticInvariant(rbf) + StochasticInvariant(deepkernel)      StochasticConvolvedInducingPoints + ConvolvedKernelSpaceInducingPoints(cnn(X)))
#---------------------------------------------------------------------------------------------------
# 2. check that a 'real' deep kernel with dimensionality reduction works (here we only check that it compiles as we have no "ground truth" available)


######## define some helpers
def generate_data():
    X = np.random.uniform(-3, 3, 400)[:, None]
    X = np.reshape(X, [200, 2])  # 2-dimensional input
    Y = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)[..., None] + np.random.randn(len(X), 1) * 0.1
    return X, Y

def key_mapper(key):
        if 'kernel' in key:
            key = key.replace('kernel', 'kernel.basekern')
        return key

# kernel makers
def rbf():
    return gpflow.kernels.SquaredExponential()

def cnn():
    cnn = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(2,)),
        tf.keras.layers.Lambda(lambda x: tf.cast(x, default_float()))]) 
    return cnn

def deepkernel():
    deepkernel = DeepKernel((2, 1),
                    filters_l1=0,
                    filters_l2=0,
                    hidden_units=0,
                    output_units=2,
                    basekern=gpflow.kernels.SquaredExponential(),
                    batch_size=200,
                    cnn=cnn())
    return deepkernel
 
def invariant_base_rbf():
    invariant_kernel = invgp.kernels.Invariant(
        basekern=gpflow.kernels.SquaredExponential(),
        orbit=invgp.kernels.orbits.SwitchXY())
    return invariant_kernel

def invariant_base_deepkernel():
    invariant_deepkernel = invgp.kernels.Invariant(
        basekern=deepkernel(),
        orbit=invgp.kernels.orbits.SwitchXY())
    return invariant_deepkernel
 
def stochasticinvariant_base_rbf():
    stochasticinvariant_kernel = invgp.kernels.StochasticInvariant(
        basekern=gpflow.kernels.SquaredExponential(),
        orbit=invgp.kernels.orbits.SwitchXY())
    return stochasticinvariant_kernel

def stochasticinvariant_base_deepkernel():
    stochasticinvariant_deepkernel = invgp.kernels.StochasticInvariant(
        basekern=deepkernel(),
        orbit=invgp.kernels.orbits.SwitchXY())
    return stochasticinvariant_deepkernel


######## define tests
@pytest.mark.parametrize("kern", [
    lambda: (rbf(), deepkernel()),
    lambda: (invariant_base_rbf(), invariant_base_deepkernel()),
    lambda: (stochasticinvariant_base_rbf(), stochasticinvariant_base_deepkernel())
])
@pytest.mark.parametrize("indp", [
    lambda X: (X.copy(), cnn()(X.copy())),
    lambda X: (ConvolvedInducingPoints(X.copy()), ConvolvedKernelSpaceInducingPoints(cnn()(X.copy()))),
    lambda X: (StochasticConvolvedInducingPoints(X.copy()), ConvolvedKernelSpaceInducingPoints(cnn()(X.copy())))
])
def test_identity_deep_kernel(kern, indp):
    print('RUNNING')
    # generate datapoints - easy dataset
    X, Y = generate_data()

    train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    train_dataset = train_dataset.shuffle(1024).batch(len(X))

    sample_SVGP_model, deepkernel_sample_SVGP_model = [
        SampleSVGP(
            kern()[i],
            gpflow.likelihoods.Gaussian(),
            inducing_variable=indp(X)[i],
            num_data=len(X)
        )
        for i in range(2)
    ]

    # check that untrained sample ELBOs are the same
    untrained_elbos = [sample_SVGP_model.elbo((X, Y)).numpy() for _ in range(100)]
    untrained_deepkernel_elbos = [deepkernel_sample_SVGP_model.elbo((X, Y)).numpy() for _ in range(100)]
    print("""Untrained elbo (mean of samples): %s, \n Untrained deepkernel elbo (mean): %s"""
        %(np.mean(untrained_elbos), np.mean(untrained_deepkernel_elbos)))
    np.testing.assert_allclose(np.mean(untrained_elbos), np.mean(untrained_deepkernel_elbos), rtol=0.05, atol=0.0)

    train_iter = iter(train_dataset.repeat())
    training_loss = sample_SVGP_model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.keras.optimizers.Adam(0.001)
    set_trainable(sample_SVGP_model.inducing_variable, False)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, sample_SVGP_model.trainable_variables)

    elbo_hist = []
    for step in range(2000):
        optimization_step()
        if step % 10 == 0:
            minibatch_elbo = -training_loss().numpy()
            print("Step: %s, Mini batch elbo: %s" % (step, minibatch_elbo))
            elbo_hist.append(minibatch_elbo)


    # initialize deepkernel SVGP model with fitted parameters from SVGP
    trained_params = read_values(sample_SVGP_model)
    trained_params = {key_mapper(key): value for key, value in trained_params.items()}

    multiple_assign(deepkernel_sample_SVGP_model, trained_params)
    gpflow.utilities.print_summary(sample_SVGP_model)
    gpflow.utilities.print_summary(deepkernel_sample_SVGP_model)

    # check that untrained sample ELBOs are the same
    trained_elbos = [sample_SVGP_model.elbo((X, Y)).numpy() for _ in range(100)]
    trained_deepkernel_elbos = [deepkernel_sample_SVGP_model.elbo((X, Y)).numpy() for _ in range(100)]
    print("""Trained elbo (mean of samples): %s, \n Trained deepkernel elbo (mean): %s"""
        %(np.mean(trained_elbos), np.mean(trained_deepkernel_elbos)))
    np.testing.assert_allclose(np.mean(trained_elbos), np.mean(trained_deepkernel_elbos), rtol=0.05, atol=0.0)


def test_dimred_deep_kernel():
    X, Y = generate_data()
    # dimensionality reduction map
    cnn = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(2,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation=None),
        tf.keras.layers.Lambda(lambda x: tf.cast(x, default_float()))])

    deepkernel = DeepKernel((2, 1),
                    filters_l1=0,
                    filters_l2=0,
                    hidden_units=0,
                    output_units=2,
                    basekern=gpflow.kernels.SquaredExponential(),
                    batch_size=200,
                    cnn=cnn)

    inducing_variables = KernelSpaceInducingPoints(deepkernel.cnn(X.copy()))


    deepkernel_sample_SVGP_model = SampleSVGP(
            deepkernel,
            gpflow.likelihoods.Gaussian(),
            inducing_variable=inducing_variables,
            num_data=len(X),
            matheron_sampler=True)

    untrained_deepkernel_elbos = [deepkernel_sample_SVGP_model.elbo((X, Y)).numpy() for _ in range(100)]
    print("Untrained elbo (mean of samples):", untrained_deepkernel_elbos.mean())
