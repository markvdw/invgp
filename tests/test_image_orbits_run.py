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
    orbits.GeneralSpatialTransform(minibatch_size=10, initialization=0.02),
    orbits.InterpretableSpatialTransform(minibatch_size=10),
    orbits.ImageRotQuant(orbit_size=10, angle=90, interpolation_method="NEAREST"),
    orbits.ImageRotQuant(orbit_size=10, angle=5, interpolation_method="NEAREST"),
])
def test_if_orbit_runs(orbit):
    (X, Y), _ = load_mnist()
    Xo = orbit(X[:2, :])

# A sligtly more advanced test that checks whether the orbit has the right number of images
@pytest.mark.parametrize("angle", [10, 30, 90])
def test_flexible_orbit_size(angle):
    (X, Y), _ = load_mnist()
    orbit = orbits.ImageRotQuant(orbit_size=10, angle=angle, interpolation_method="NEAREST")
    Xo = orbit(X[:1, :])
    orbit_size = Xo.shape[1]
    assert orbit_size == 10
