"""
Simply test that the image orbits run...
"""
import pytest

from invgp.kernels import orbits
from invgp_experiments.datasets import load_mnist


@pytest.mark.parametrize("orbit", [
    orbits.ImageRotation(90, minibatch_size=10, interpolation_method="NEAREST"),
    orbits.ImageRotation(90, minibatch_size=10, interpolation_method="BILINEAR"),
    orbits.ImageRotation(90, minibatch_size=10, interpolation_method="BILINEAR", use_stn=True),
    orbits.GeneralSpatialTransform(minibatch_size=10)
])
def test_if_orbit_runs(orbit):
    (X, Y), _ = load_mnist()
    Xo = orbit(X[:2, :])
