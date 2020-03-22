"""
Simply test that the image orbits run...
"""
import matplotlib.pyplot as plt
import pytest

from invgp.kernels import orbits
from invgp_experiments.datasets import load_mnist

# (X, Y), _ = load_mnist()
#
# o = orbits.ImageRotation(90, minibatch_size=10, interpolation_method="BILINEAR", use_stn=True)
# Xo = o(X[:2, :])
# plt.imshow(Xo[0, 1, :].numpy().reshape(28, 28))


@pytest.mark.parametrize("orbit", [
    orbits.ImageRotation(90, minibatch_size=10, interpolation_method="NEAREST"),
    orbits.ImageRotation(90, minibatch_size=10, interpolation_method="BILINEAR"),
    orbits.ImageRotation(90, minibatch_size=10, interpolation_method="BILINEAR", use_stn=True)
])
def test_if_orbit_runs(orbit):
    (X, Y), _ = load_mnist()
    Xo = orbit(X[:2, :])
