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


######## define tests
# Test: use deep kernel with identity function to see that it's the same as the non deep kernel
def test_identity_deep_kernel():
    # generate datapoints - easy dataset
    X, Y = generate_data()

    # generate datapoints - hard dataset
    # X = np.random.uniform(-3, 3, 400)[:, None]
    # X = np.reshape(X, [200, 2])  # 2-dimensional input
    # Y = np.ones((200, 1))
    # Y[np.where(np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2) > 2)] = 0

    train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    train_dataset = train_dataset.shuffle(1024).batch(len(X))

    # plot everything
    fig, ax = plt.subplots(1, 1, figsize=(12, 15))
    x1list = np.linspace(-3.0, 3.0, 100)
    x2list = np.linspace(-3.0, 3.0, 100)
    X1, X2 = np.meshgrid(x1list, x2list)
    ax.set_title('True function')
    ax.set_aspect('equal', 'box')
    # plot the true data generating process
    #true_Y = np.ones_like(X1)
    #true_np.sqrt(X1 ** 2 + X2 ** 2) > 2)] = 0
    true_Y = np.sqrt(X1 ** 2 + X2 ** 2)
    cp = ax.contourf(X1, X2, true_Y)
    plt.colorbar(cp)
    plt.show()

    # initialize SVGP model
    inducing_variables = X.copy()

    kernel = gpflow.kernels.SquaredExponential()

    invariant_kernel = invgp.kernels.StochasticInvariant(
        basekern=gpflow.kernels.SquaredExponential(), # or the deepkernel
        orbit=invgp.kernels.orbits.SwitchXY())

    sample_SVGP_model = SampleSVGP(
            kernel,
            gpflow.likelihoods.Gaussian(),
            inducing_variable=inducing_variables,
            num_data=len(X),
            matheron_sampler=True)

    # initialize DeepKernel models
    # idenitity map
    cnn = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(2,)),
        tf.keras.layers.Lambda(lambda x: tf.cast(x, default_float()))])  # input_shape)])

    deepkernel = DeepKernel((2, 1),
                    filters_l1=0,
                    filters_l2=0,
                    hidden_units=0,
                    output_units=2,
                    basekern=gpflow.kernels.SquaredExponential(),
                    batch_size=200,
                    cnn=cnn)


    deepkernel_sample_SVGP_model = SampleSVGP(
            deepkernel,
            gpflow.likelihoods.Gaussian(),
            inducing_variable=inducing_variables,
            num_data=len(X),
            matheron_sampler=True)

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

    # TODO, other tests:
    # 1. check that dim red. deep kernel also works -DONE
    # 2. check invariant
    # 3. train 'real' deep kernel to see that it can fit the data better

def test_identity_invariant_deep_kernel():
    # generate datapoints - easy dataset
    X, Y = generate_data()
    train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    train_dataset = train_dataset.shuffle(1024).batch(len(X))

    # initialize invariant kernel
    inducing_variables = ConvolvedInducingPoints(X.copy())
    invariant_kernel = invgp.kernels.Invariant(
        basekern=gpflow.kernels.SquaredExponential(), # or the deepkernel
        orbit=invgp.kernels.orbits.SwitchXY())


    # initialise invariant deep kernel
    # idenitity map
    cnn = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(2,)),
        tf.keras.layers.Lambda(lambda x: tf.cast(x, default_float()))])  # input_shape)])
    deepkernel = DeepKernel((2, 1),
                    filters_l1=0,
                    filters_l2=0,
                    hidden_units=0,
                    output_units=2,
                    basekern=gpflow.kernels.SquaredExponential(),
                    batch_size=200,
                    cnn=cnn)
    invariant_deepkernel = invgp.kernels.Invariant( # or Stochastic ...
        basekern=deepkernel,
        orbit=invgp.kernels.orbits.SwitchXY())

    # initialize invariant models
    sample_SVGP_model = SampleSVGP(
            invariant_kernel,
            gpflow.likelihoods.Gaussian(),
            inducing_variable=inducing_variables,
            num_data=len(X),
            matheron_sampler=True)

    # or Stochastic ...
    kernel_space_inducing_variables = ConvolvedKernelSpaceInducingPoints(deepkernel.cnn(X.copy()))
    deepkernel_sample_SVGP_model = SampleSVGP(
            invariant_deepkernel,
            gpflow.likelihoods.Gaussian(),
            inducing_variable=kernel_space_inducing_variables,
            num_data=len(X),
            matheron_sampler=True)

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




if __name__=='__main__':
    # test_identity_deep_kernel()
    test_identity_invariant_deep_kernel()
    # test_dimred_deep_kernel()
