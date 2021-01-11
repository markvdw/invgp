"""
Simply test that the image orbits run...
"""
import pytest

from invgp.kernels import orbits
from deepkernelinv.utils.load_datasets import load
import numpy as np
import tensorflow_datasets as tfds


class args:
    dataset = "CIFAR10"
    image_shape = (32, 32, 3) if dataset == "CIFAR10" else (28, 28)
    subset_size = None


def load_cifar():
    train, test = load(args)
    X, _ = list(zip(*[a for a in tfds.as_numpy(train.take(7))]))
    X = np.stack(X).astype('float64')
    return X


@pytest.mark.parametrize("orbit", [
    orbits.GeneralSpatialTransform(minibatch_size=10, initialization=0.02, colour=True),
    orbits.InterpretableSpatialTransform(minibatch_size=10, colour=True),
    orbits.ColorTransform(minibatch_size=10, log_lims_contrast=[-1., 1.], log_lims_brightness=[-1., 1.])
])
def test_if_orbit_runs(orbit):
    X = load_cifar()
    Xo = orbit(X[:2, :])
