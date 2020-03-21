"""
This test needs a bit of explanation. It may be a bit overkill, but I think
it's nice. For an orbit which adds Gaussian noise, we can compute all the
resutling covariances. They're simply Gaussian! This is similar to the
Multiscale interdomain feature in GPflow.

Here we do a simple implementation of the closed-form computations, so we can
directly check whether the stochastic method has an unbiased estimate of the
ELBO.
"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import gpflow
import invgp
from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.kernels import SquaredExponential
from gpflow.utilities import set_trainable
from gpflow.utilities.ops import square_distance
from invgp.kernels import orbits


class GaussianNoiseInvariant(SquaredExponential):
    def __init__(self, base_variance=1.0, base_lengthscales=1.0, orbit_variance=1.0):
        self.base_variance = gpflow.Parameter(base_variance, transform=gpflow.utilities.bijectors.positive())
        self.base_lengthscales = gpflow.Parameter(base_lengthscales, transform=gpflow.utilities.bijectors.positive())
        self.orbit_variance = orbit_variance
        super().__init__()

    @property
    def lengthscales(self):
        return (self.base_lengthscales ** 2.0 + 2 * self.orbit_variance) ** 0.5

    @lengthscales.setter
    def lengthscales(self, x):
        pass

    @property
    def variance(self):
        return (self.base_lengthscales ** 2.0 / self.lengthscales ** 2.0) ** 0.5 * self.base_variance

    @variance.setter
    def variance(self, x):
        pass


class GaussianNoiseInvariantInducing(gpflow.inducing_variables.InducingPoints):
    pass


@Kuu.register(GaussianNoiseInvariantInducing, GaussianNoiseInvariant)
def Kuu_temp(inducing_variable, kernel, *, jitter=0.0):
    print("Kuu_temp")
    X_scaled = inducing_variable.Z / kernel.base_lengthscales
    r2 = square_distance(X_scaled, None)
    return kernel.base_variance * tf.exp(-0.5 * r2) + jitter * tf.eye(len(inducing_variable), dtype=r2.dtype)


@Kuf.register(GaussianNoiseInvariantInducing, GaussianNoiseInvariant, object)
def Kuf_temp(inducing_variable, kernel, Xnew):
    print("Kuf_temp")
    len = (kernel.base_lengthscales ** 2.0 + kernel.orbit_variance) ** 0.5
    X_scaled = Xnew / len
    Z_scaled = inducing_variable.Z / len
    r2 = square_distance(Z_scaled, X_scaled)
    return kernel.base_variance * (kernel.base_lengthscales ** 2.0 / len ** 2.0) ** 0.5 * tf.exp(-0.5 * r2)


def test_elbo():
    np.random.seed(0)
    Z = np.linspace(-0.5, 6.5, 20)[:, None]

    # Set up kernels
    base_var = 0.73
    base_len = 1.5
    orbit_variance = 0.5

    stoch_k = invgp.kernels.StochasticInvariant(SquaredExponential(variance=base_var, lengthscales=base_len),
                                                orbits.GaussianNoiseOrbit(variance=orbit_variance))
    stoch_k.orbit.minibatch_size = 100
    det_k = GaussianNoiseInvariant(base_var, base_len, orbit_variance)

    X = np.linspace(-5, 5, 100)[:, None]
    X2 = np.array([[0.0]])
    stoch_kernfunc = sum([stoch_k.K(X, X2) for _ in tqdm(range(100))]) / 100
    det_kernfunc = det_k.K(X, X2)
    np.testing.assert_allclose(stoch_kernfunc, det_kernfunc, rtol=0.01, atol=1e-2)

    stoch_inducing_variable = invgp.inducing_variables.StochasticConvolvedInducingPoints(Z.copy())
    det_inducing_variable = GaussianNoiseInvariantInducing(Z.copy())

    stoch_Kzz = Kuu(stoch_inducing_variable, stoch_k)
    det_Kzz = Kuu(det_inducing_variable, det_k)
    np.testing.assert_allclose(stoch_Kzz, det_Kzz)

    stoch_Kzx = sum([tf.reduce_mean(Kuf(stoch_inducing_variable, stoch_k, X), -1) for _ in tqdm(range(100))]) / 100
    det_Kzx = Kuf(det_inducing_variable, det_k, X)
    np.testing.assert_allclose(stoch_Kzx, det_Kzx, rtol=0.01, atol=1e-2)

    # Generate data
    X = np.random.uniform(0, 6, 200)[:, None]
    Y = np.sin(2 * X) + 0.1 * np.cos(7 * X) + np.random.randn(*X.shape) * 0.1

    # Train deterministic model
    det_m = gpflow.models.SGPR((X, Y), det_k, inducing_variable=det_inducing_variable)
    set_trainable(det_m.inducing_variable, False)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(tf.function(lambda: -det_m.log_marginal_likelihood()), det_m.trainable_variables,
                 options=dict(maxiter=1000))
    q_mu, q_var = [_.numpy() for _ in det_m.compute_qu()]

    # Train stochastic model
    stoch_m = gpflow.models.SVGP(stoch_k, gpflow.likelihoods.Gaussian(), stoch_inducing_variable, whiten=False)
    stoch_m.kernel.basekern.lengthscales.assign(det_m.kernel.base_lengthscales.numpy())
    stoch_m.kernel.basekern.variance.assign(det_m.kernel.base_variance.numpy())
    stoch_m.likelihood.variance.assign(det_m.likelihood.variance.numpy())
    stoch_m.q_mu.assign(q_mu)
    stoch_m.q_sqrt.assign(np.linalg.cholesky(q_var)[None, :, :])

    stoch_m.log_marginal_likelihood((X, Y))
    fast_lml = tf.function(lambda: stoch_m.log_marginal_likelihood((X, Y)))
    stoch_lml = np.mean([fast_lml() for _ in tqdm(range(1000))])
    det_lml = det_m.log_marginal_likelihood()

    print(stoch_lml)
    print(det_lml)

    np.testing.assert_allclose(stoch_lml, det_lml, rtol=0.05, atol=0.0)
