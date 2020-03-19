import tensorflow as tf

from gpflow.base import TensorLike
from gpflow.covariances.dispatch import Kuf, Kuu
from ..inducing_variables import ConvolvedInducingPoints
from ..kernels import Invariant


@Kuu.register(ConvolvedInducingPoints, Invariant)
def Kuu_convip_invariant(inducing_variable: ConvolvedInducingPoints, kernel: Invariant, *, jitter=0.0):
    Kzz = kernel.basekern.K(inducing_variable.Z)
    Kzz += jitter * tf.eye(len(inducing_variable), dtype=Kzz.dtype)
    return Kzz


@Kuf.register(ConvolvedInducingPoints, Invariant, TensorLike)
def Kuf_convip_invariant(inducing_variable, kern, Xnew):
    N, M = tf.shape(Xnew)[0], tf.shape(inducing_variable.Z)[0]
    orbit_size = kern.orbit.orbit_size
    Xorbit = kern.orbit(Xnew)  # [N, orbit_size, D]
    Kzx_orbit = kern.basekern.K(inducing_variable.Z, tf.reshape(Xorbit, (N * orbit_size, -1)))  # [M, N * orbit_sz]
    Kzx = tf.reduce_mean(tf.reshape(Kzx_orbit, (M, N, orbit_size)), [2])
    return Kzx
