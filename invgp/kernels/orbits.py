import numpy as np
import tensorflow as tf
from tensorflow_probability import bijectors as tfb

import gpflow
from gpflow.config import default_float
from gpflow.utilities.bijectors import positive
from .image_transforms import rotate_img_angles, rotate_img_angles_stn, apply_stn_batch


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

    def __init__(self, rotation_quantisation=45, angle=360, interpolation_method="NEAREST", input_dim=None, img_size=None,
                 use_stn=False, **kwargs):
        super().__init__(int(angle / rotation_quantisation), input_dim=input_dim, img_size=img_size, **kwargs)
        self.interpolation = interpolation_method if not use_stn else "BILINEAR"
        self.angle = gpflow.Parameter(angle)
        self.rotation_quantisation = rotation_quantisation
        self.interpolation_method = interpolation_method
        self.angles = np.arange(0, self.angle, rotation_quantisation)
        self.angles = np.arange(0, self.angle.numpy(), rotation_quantisation)
        self.use_stn = use_stn

    def orbit_full(self, X):
        img_size = self.img_size(X)
        Ximgs = tf.reshape(X, [-1, img_size, img_size])
        if self.use_stn:
            return rotate_img_angles_stn(Ximgs, self.angles)  # STN always uses bilinear interpolation
        else:
            return rotate_img_angles(Ximgs, self.angles, self.interpolation)


ANGLE_JITTER = 1e0  # minimal value for the angle variable (to be safe when transforming to logistic)


class ImageRotation(ImageOrbit):
    def __init__(self, angle=ANGLE_JITTER, interpolation_method="NEAREST", use_stn=False,
                 input_dim=None, img_size=None, minibatch_size=10, **kwargs):
        super().__init__(np.inf, input_dim=input_dim, img_size=img_size, minibatch_size=minibatch_size, **kwargs)
        self.interpolation = interpolation_method if not use_stn else "BILINEAR"
        low_const = tf.constant(0.0, dtype=default_float())
        high_const = tf.constant(180.0, dtype=default_float())
        self.angle = gpflow.Parameter(angle, transform=tfb.Sigmoid(low_const, high_const))  # constrained to [0, 180]
        self.use_stn = use_stn

    def orbit_minibatch(self, X):
        # Reparameterise angle
        eps = tf.random.uniform([self.minibatch_size], 0., 1., dtype=default_float())
        angles = -self.angle + 2. * self.angle * eps
        Ximgs = tf.reshape(X, [-1, self.img_size(X), self.img_size(X)])
        if self.use_stn:
            return rotate_img_angles_stn(Ximgs, angles)  # STN always uses bilinear interpolation
        else:
            return rotate_img_angles(Ximgs, angles, self.interpolation)


class GeneralSpatialTransform(ImageOrbit):
    """
    Kernel invariant to to transformations using Spatial Transformer Networks (STNs); this correponds to six-parameter
    affine transformations.
    This version of the kernel is parameterised by the six independent parameters directly (thus "_general")
    """

    def __init__(self, theta_min=np.array([1., 0., 0., 0., 1., 0.]),
                 theta_max=np.array([1., 0., 0., 0., 1., 0.]), constrain=False, input_dim=None, img_size=None,
                 minibatch_size=10, **kwargs):
        """
        :param theta_min: one end of the range; identity = [1, 0, 0, 0, 1, 0]
        :param theta_max: other end of the range; identity = [1, 0, 0, 0, 1, 0]
        :param constrain: whether theta_min is always below the identity and theta_max always above
        """
        super().__init__(np.inf, input_dim=input_dim, img_size=img_size, minibatch_size=minibatch_size, **kwargs)
        self.constrain = constrain
        if constrain:
            self.theta_min_0 = gpflow.Parameter(1. - theta_min[0], dtype=default_float(), transform=positive())
            self.theta_min_1 = gpflow.Parameter(-theta_min[1], dtype=default_float(), transform=positive())
            self.theta_min_2 = gpflow.Parameter(-theta_min[2], dtype=default_float(), transform=positive())
            self.theta_min_3 = gpflow.Parameter(-theta_min[3], dtype=default_float(), transform=positive())
            self.theta_min_4 = gpflow.Parameter(1. - theta_min[4], dtype=default_float(), transform=positive())
            self.theta_min_5 = gpflow.Parameter(-theta_min[5], dtype=default_float(), transform=positive())

            self.theta_max_0 = gpflow.Parameter(theta_min[0], dtype=default_float(), transform=positive(lower=1.))
            self.theta_max_1 = gpflow.Parameter(theta_min[1], dtype=default_float(), transform=positive())
            self.theta_max_2 = gpflow.Parameter(theta_min[2], dtype=default_float(), transform=positive())
            self.theta_max_3 = gpflow.Parameter(theta_min[3], dtype=default_float(), transform=positive())
            self.theta_max_4 = gpflow.Parameter(theta_min[4], dtype=default_float(), transform=positive(lower=1.))
            self.theta_max_5 = gpflow.Parameter(theta_min[5], dtype=default_float(), transform=positive())
        else:
            self.theta_min = gpflow.Parameter(theta_min, dtype=default_float())
            self.theta_max = gpflow.Parameter(theta_max, dtype=default_float())

    def orbit_minibatch(self, X):
        eps = tf.random.uniform([self.minibatch_size, 6], 0., 1., dtype=default_float())
        if self.constrain:
            theta_min = tf.stack([1. - self.theta_min_0, -self.theta_min_1, -self.theta_min_2, -self.theta_min_3,
                                  1. - self.theta_min_4, -self.theta_min_5])
            theta_max = tf.stack([self.theta_max_0, self.theta_max_1, self.theta_max_2, self.theta_max_3,
                                  self.theta_max_4, self.theta_max_5])
            theta_min = tf.reshape(theta_min, [1, -1])
            theta_max = tf.reshape(theta_max, [1, -1])
        else:
            theta_min = tf.reshape(self.theta_min, [1, -1])
            theta_max = tf.reshape(self.theta_max, [1, -1])
        thetas = theta_min + (theta_max - theta_min) * eps

        Ximgs = tf.reshape(X, [-1, self.img_size(X), self.img_size(X)])

        return apply_stn_batch(Ximgs, thetas)
