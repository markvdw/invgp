import numpy as np
import tensorflow as tf
import gpflow
import invgp
from invgp_experiments import datasets
import matplotlib.pyplot as plt
import numpy.random as rnd
from invgp.inducing_variables.invariant_convolution_domain import StochasticConvolvedInducingPoints
from invgp.models import sample_SVGP
from gpflow.models import SVGP

import numpy as np
import tensorflow as tf


np.random.seed(1)

# Generate dataset
(X, Y), _ = datasets.load_mnist(digits=[2, 3])
X = X[:50, :]
Y = Y[:50, 0]
Y = tf.one_hot(tf.cast(Y, tf.uint8), 10)
Y = tf.cast(Y, tf.float64)
print('data shapes are', X.shape, Y.shape)
print(X.shape, Y.shape)

# initialize models
nr_inducing_points = 100
inducing_variables = X[rnd.permutation(len(X))[:nr_inducing_points], :]
inducing_variables = StochasticConvolvedInducingPoints(inducing_variables)
basekernel = gpflow.kernels.SquaredExponential()
orbit = invgp.kernels.orbits.ImageRotation(
            input_dim=(28 * 28),
            minibatch_size=100,
            use_stn=True)
kernel = invgp.kernels.StochasticInvariant(
                 basekern=basekernel,
                 orbit=orbit)
likelihood = gpflow.likelihoods.Gaussian()

sample_SVGP_model = sample_SVGP.sample_SVGP(kernel, likelihood,
                               inducing_variable=inducing_variables,
                               num_data=50,
                               num_latent_gps=10)
SVGP_model = SVGP(kernel, likelihood,
                               inducing_variable=inducing_variables,
                               num_data=50,
                               num_latent_gps=10)

# compute samples
sample_elbo = sample_SVGP_model.elbo((X, Y))
elbo = SVGP_model.elbo((X, Y))

print(' sample_elbo:', sample_elbo.numpy(), '\n elbo:', elbo.numpy())
