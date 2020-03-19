import numpy as np
import tensorflow as tf

import gpflow
from .image_transforms import rotate_img_angles


class Invariant(gpflow.kernels.Kernel):
    def __init__(self, basekern, orbit):
        super().__init__()
        self.basekern = basekern
        self.orbit = orbit

    def K(self, X, X2=None):
        X_orbit = self.orbit(X)
        N, num_orbit_points, D = tf.shape(X_orbit)[0], tf.shape(X_orbit)[1], tf.shape(X)[1]
        X_orbit = tf.reshape(X_orbit, (-1, D))
        Xp2 = tf.reshape(self.orbit(X2), (-1, D)) if X2 is not None else None

        bigK = self.basekern.K(X_orbit, Xp2)  # [N * num_patches, N * num_patches]
        K = tf.reduce_mean(tf.reshape(bigK, (N, num_orbit_points, -1, num_orbit_points)), [1, 3])
        return K

    def K_diag(self, X):
        Xp = self.orbit(X)

        def sumbK(Xp):
            return tf.reduce_mean(self.basekern.K(Xp))

        # Can use vectorised_map?
        return tf.map_fn(sumbK, Xp)
        # return tf.reduce_sum(tf.map_fn(self.basekern.K, Xp), [1, 2]) / self.num_patches ** 2.0








