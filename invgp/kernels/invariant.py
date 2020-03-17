import tensorflow as tf

import gpflow


class InvariantBase(gpflow.kernels.Kernel):
    def orbit(self, X):
        raise NotImplementedError

    @property
    def orbit_size(self):
        return self._orbit_size


class Invariant(InvariantBase):
    def __init__(self, basekern, orbit_size):
        super().__init__()
        self.basekern = basekern
        self._orbit_size = orbit_size

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


class SwitchDimsInvariant(Invariant):
    def __init__(self, basekern):
        super().__init__(basekern, orbit_size=2)

    def orbit(self, X):
        X_switch = tf.gather(X, [1, 0], axis=1)
        return tf.concat([X[:, None, :], X_switch[:, None, :]], axis=1)
