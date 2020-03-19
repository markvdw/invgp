import numpy as np
import pytest

import gpflow
from gpflow.config import default_jitter
from invgp.covariances import Kuu, Kuf
from invgp.inducing_variables import ConvolvedInducingPoints
from invgp.kernels import Invariant, orbits


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
