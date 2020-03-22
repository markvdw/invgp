import numpy as np
import tensorflow as tf
from tensorflow_probability import bijectors as tfb

import gpflow
from .image_transforms import rotate_img_angles, rotate_img_angles_stn


class Orbit(gpflow.base.Module):
    def __init__(self, orbit_size, minibatch_size=None, name=None):
        super().__init__(name=name)
        self._orbit_size = orbit_size
        self.minibatch_size = minibatch_size if minibatch_size is not None else orbit_size

    @property
    def orbit_size(self):
        return self._orbit_size

    def orbit_full(self, X):
        raise NotImplementedError

    def orbit_minibatch(self, X):
        full_orbit = tf.transpose(self.orbit_full(X), [1, 0, 2])  # [orbit_size, X.shape[0], ...]
        return tf.transpose(tf.random.shuffle(full_orbit)[:self.minibatch_size, :, :], [1, 0, 2])

    def __call__(self, X):
        if self.minibatch_size == self.orbit_size:
            return self.orbit_full(X)
        else:
            return self.orbit_minibatch(X)


class SwitchXY(Orbit):
    def __init__(self, **kwargs):
        super().__init__(2, **kwargs)

    def orbit_full(self, X):
        X_switch = tf.gather(X, [1, 0], axis=1)
        return tf.concat([X[:, None, :], X_switch[:, None, :]], axis=1)


class GaussianNoiseOrbit(Orbit):
    def __init__(self, variance=1.0, minibatch_size=10, **kwargs):
        super().__init__(np.inf, minibatch_size, **kwargs)
        self.variance = variance

    def orbit_minibatch(self, X):
        return X[:, None, :] + tf.random.normal((X.shape[0], self.minibatch_size, X.shape[1]),
                                                stddev=self.variance ** 0.5, dtype=X.dtype)


class ImageOrbit(Orbit):
    def __init__(self, orbit_size, input_dim=None, img_size=None, minibatch_size=None, **kwargs):
        super().__init__(orbit_size, minibatch_size=minibatch_size, **kwargs)
        if input_dim is not None and img_size is None:
            img_size = int(input_dim ** 0.5)
        elif input_dim is None and img_size is not None:
            input_dim = img_size ** 2
        elif input_dim is not None and img_size is not None:
            assert self._img_size ** 2 == self._input_dim
        self._img_size = img_size
        self._input_dim = input_dim

    def input_dim(self, X):
        # X can be None if not required
        if self._input_dim is not None:
            return self._input_dim
        else:
            return tf.shape(X)[1]

    def img_size(self, X):
        # X can be None if not required
        if self._img_size is not None:
            return self._img_size
        else:
            return tf.cast(tf.cast(self.input_dim(X), tf.float32) ** 0.5, tf.int32)


class ImageRot90(ImageOrbit):
    """
    ImageRot90
    Kernel invariant to 90 degree rotations of the input image.
    """

    def __init__(self, input_dim=None, img_size=None, **kwargs):
        super().__init__(4, input_dim=input_dim, img_size=img_size, **kwargs)

    def orbit_full(self, X):
        Ximgs = tf.reshape(X, [-1, self.img_size(X), self.img_size(X)])
        cc90 = tf.reshape(tf.transpose(tf.reverse(Ximgs, [-1]), [0, 2, 1]), (-1, self.input_dim(X)))
        cc180 = tf.reshape(tf.reverse(Ximgs, [-2, -1]), (-1, self.input_dim(X)))
        cc270 = tf.reshape(tf.reverse(tf.transpose(Ximgs, [0, 2, 1]), [-1]), (-1, self.input_dim(X)))
        return tf.concat((X[:, None, :], cc90[:, None, :], cc180[:, None, :], cc270[:, None, :]), 1)


class ImageRotQuant(ImageOrbit):
    """
    ImageRotQuant
    Kernel invariant to any quantised rotations of the input image.
    """

    def __init__(self, rotation_quantisation=45, interpolation_method="NEAREST", input_dim=None, img_size=None,
                 **kwargs):
        super().__init__(int(360 / rotation_quantisation), input_dim=input_dim, img_size=img_size, **kwargs)
        self.rotation_quantisation = rotation_quantisation
        self.interpolation_method = interpolation_method
        assert 360 % rotation_quantisation == 0, "Orbit must complete in 360 degrees."  # Not strictly necessary
        self.angles = np.arange(0, 360, rotation_quantisation)

    def orbit_full(self, X):
        img_size = self.img_size(X)
        Ximgs = tf.reshape(X, [-1, img_size, img_size])
        return rotate_img_angles(Ximgs, self.angles, self.interpolation_method)


ANGLE_JITTER = 1e0  # minimal value for the angle variable (to be safe when transforming to logistic)


class ImageRotation(ImageOrbit):
    def __init__(self, angle=ANGLE_JITTER, interpolation_method="NEAREST", use_stn=False,
                 input_dim=None, img_size=None, minibatch_size=10, **kwargs):
        super().__init__(np.inf, input_dim=input_dim, img_size=img_size, minibatch_size=minibatch_size, **kwargs)
        self.interpolation = interpolation_method if not use_stn else "BILINEAR"
        low_const = tf.constant(0.0, dtype=gpflow.config.default_float())
        high_const = tf.constant(180.0, dtype=gpflow.config.default_float())
        self.angle = gpflow.Parameter(angle, transform=tfb.Sigmoid(low_const, high_const))  # constrained to [0, 180]
        self.use_stn = use_stn

    def orbit_minibatch(self, X):
        # Reparameterise angle
        eps = tf.random.uniform([self.minibatch_size], 0., 1., dtype=gpflow.config.default_float())
        angles = -self.angle + 2. * self.angle * eps

        Ximgs = tf.reshape(X, [-1, self.img_size(X), self.img_size(X)])
        if self.use_stn:
            return rotate_img_angles_stn(Ximgs, angles)  # STN always uses bilinear interpolation
        else:
            return rotate_img_angles(Ximgs, angles, self.interpolation)
