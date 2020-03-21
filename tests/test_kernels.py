import numpy as np
import pytest
from tqdm import tqdm

import gpflow
from gpflow.config import default_jitter
from invgp.covariances import Kuu, Kuf
from invgp.inducing_variables import ConvolvedInducingPoints
from invgp.kernels import Invariant, StochasticInvariant, orbits
from invgp_experiments.datasets import load_mnist


def test_conv_diag():
    kernel = Invariant(gpflow.kernels.SquaredExponential(), orbits.SwitchXY())
    X = np.random.randn(3, 2)
    kernel_full = np.diagonal(kernel(X, full_cov=True))
    kernel_diag = kernel(X, full_cov=False)
    assert np.allclose(kernel_full, kernel_diag)


_inducing_variables_and_kernels = [
    [
        2,
        ConvolvedInducingPoints(np.random.randn(71, 2)),
        Invariant(gpflow.kernels.SquaredExponential(), orbits.SwitchXY()),
    ],
    [
        4,
        ConvolvedInducingPoints(np.random.randn(71, 4)),
        Invariant(gpflow.kernels.SquaredExponential(), orbits.ImageRot90()),
    ],
]


@pytest.mark.parametrize("input_dim, inducing_variable, kernel", _inducing_variables_and_kernels)
def test_inducing_variables_psd_schur(input_dim, inducing_variable, kernel):
    # Conditional variance must be PSD.
    X = np.random.randn(5, input_dim)
    Kuf_values = Kuf(inducing_variable, kernel, X)
    Kuu_values = Kuu(inducing_variable, kernel, jitter=default_jitter())
    Kff_values = kernel(X)
    Qff_values = Kuf_values.numpy().T @ np.linalg.solve(Kuu_values, Kuf_values)
    assert np.all(np.linalg.eig(Kff_values - Qff_values)[0] > 0.0)


@pytest.mark.parametrize("orbit, orbit_kwargs, orbit_minibatch", [
    # (orbits.ImageRot90, {}, 2),  # Requires many samples for orbit_minibatch of 2
    (orbits.ImageRotQuant, dict(interpolation_method="BILINEAR"), 6)
])
def test_unbiased_kernel_estimate(orbit, orbit_kwargs, orbit_minibatch):
    det_k = Invariant(gpflow.kernels.SquaredExponential(lengthscales=5), orbit(**orbit_kwargs))
    stoch_k = StochasticInvariant(gpflow.kernels.SquaredExponential(lengthscales=5), orbit(**orbit_kwargs))
    (X, Y), _ = load_mnist(digits=[2, 3])
    X = X[:2, :]
    X = det_k.orbit(X).numpy().reshape(-1, X.shape[1])

    # First check if equal when minibatch_size == orbit_size
    det_K = det_k.K(X).numpy()
    stoch_K = stoch_k.K(X).numpy()

    np.testing.assert_allclose(det_K, stoch_K)

    # Check unbiased estimate
    stoch_k.orbit.minibatch_size = orbit_minibatch
    det_K = det_k.K(X)
    stoch_K_evals = [stoch_k.K(X).numpy() for _ in tqdm(range(200))]
    stoch_K = sum(stoch_K_evals) / len(stoch_K_evals)

    np.testing.assert_allclose(stoch_K, det_K, rtol=0.01)

    # Check diagonal
    stoch_K_evals = [stoch_k.K_diag(X).numpy() for _ in tqdm(range(200))]
    stoch_K = sum(stoch_K_evals) / len(stoch_K_evals)

    np.testing.assert_allclose(stoch_K, np.diag(det_K), rtol=0.01, atol=1e-3)


def test_infinite_orbit_unbiased_kernel_estimate():
    k = StochasticInvariant(gpflow.kernels.SquaredExponential(),
                            orbits.GaussianNoiseOrbit(variance=0.2, minibatch_size=100))
    newlen_squared = (k.basekern.lengthscales + 2 * k.orbit.variance).numpy()
    comp_k = gpflow.kernels.SquaredExponential(lengthscales=newlen_squared ** 0.5,
                                               variance=np.sqrt(k.basekern.lengthscales / newlen_squared))

    X1 = np.random.randn(100, 1)
    X2 = np.random.randn(2, 1)

    repeats = 500
    stoch_K = sum([k.K(X1, X2) for _ in tqdm(range(repeats))]) / repeats
    comp_K = comp_k.K(X1, X2)

    np.testing.assert_allclose(stoch_K, comp_K, rtol=0.01, atol=1e-3)
