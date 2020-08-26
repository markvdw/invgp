import tensorflow as tf
import numpy as np
from invgp.kernels import Invariant
from gpflow.utilities import Dispatcher
from gpflow.kernels import Kernel
from gpflow import kernels, covariances, default_jitter

sample_matheron = Dispatcher("sample_matheron")

@sample_matheron.register(object, object, Invariant, object, object)
def _sample_matheron(Xnew, inducing_variable, kernel, q_mu, q_sqrt, white = True, num_samples = 1, num_basis = 1024):
    X_o = kernel.orbit(Xnew)
    print('X_o shape is ', X_o.shape)
    Xnew = tf.reshape(X_o, [-1, X_o.shape[2]]) # [NS_o, D]
    # Xnew = tf.reshape(X_o, [X_o.shape[0]*X_o.shape[1], X_o.shape[2]]) # [NS_o, D]
    samples = sample_matheron(Xnew,  inducing_variable, kernel.basekern, q_mu, q_sqrt, white = white, num_samples=num_samples, num_basis=num_basis)
    samples = tf.reshape(samples, [num_samples, -1, X_o.shape[1], samples.shape[2]]) # [S, N, C, P]
    # samples = tf.reshape(samples, [num_samples, X_o.shape[0], X_o.shape[1], samples.shape[2]]) # [S, N, C, P]
    samples = tf.reduce_mean(samples, axis = 2) # [S, N, P]
    return samples

@sample_matheron.register(object, object, Kernel, object, object)
def __sample_matheron(Xnew, inducing_variable, kernel, q_mu, q_sqrt, white = True, num_samples = 1, num_basis = 1024):

    if not isinstance(kernel, kernels.SquaredExponential):
        raise NotImplementedError

    print('Running Matheron sampler.')
    # Draw from prior
    bias = tf.random.uniform(shape = [num_basis], maxval=2*np.pi, dtype = Xnew.dtype)
    spectrum = tf.random.normal(shape = [num_basis, Xnew.shape[1]], dtype = Xnew.dtype)
    _Xnew = tf.divide(Xnew,kernel.lengthscales)
    _Z = tf.divide(inducing_variable.Z, kernel.lengthscales)

    w = tf.random.normal(shape = [num_samples, q_mu.shape[1], num_basis, 1], dtype = Xnew.dtype) # [S, P, num_basis, 1]
    phiX = tf.sqrt(kernel.variance * 2 / num_basis) * tf.cos(tf.matmul(_Xnew, spectrum, transpose_b = True) + bias) # [N, num_basis]
    fX = tf.tile(phiX[None,None,:,:], [num_samples, q_mu.shape[1], 1,1] ) @ w  # [S, P, N, 1]
    phiZ = tf.sqrt(kernel.variance * 2 / num_basis) * tf.cos(tf.matmul(_Z, spectrum, transpose_b = True) + bias) # [M, num_basis]
    fZ = tf.tile(phiZ[None,None,:,:], [num_samples, q_mu.shape[1], 1,1] ) @ w   # [S, P, M, 1]
    fZ = fZ + tf.sqrt(tf.cast(default_jitter(), tf.float64)) * tf.random.normal(shape=fZ.shape, dtype=fZ.dtype) # for stability
    # Update
    eps = tf.random.normal(shape = [num_samples, q_mu.shape[1], q_mu.shape[0], 1], dtype = Xnew.dtype) # [S, P, M, 1]
    _u = tf.tile(tf.linalg.adjoint(q_mu)[None, :, :, None], [num_samples, 1,1,1]) +  tf.matmul(tf.tile(q_sqrt[None,:,:,:], [num_samples,1,1,1]), eps)# [S, P, M, 1]

    Luu = tf.linalg.cholesky(covariances.Kuu(inducing_variable, kernel, jitter=default_jitter())) # [M, M]
    Luu = tf.tile(Luu[None, None, :, :], [num_samples, q_mu.shape[1],1,1]) # [S, P, M, M]
    if white:
        _u = Luu @ _u

    res_upd = tf.linalg.cholesky_solve(Luu, _u - fZ) # [S, P, M, 1]
    f_upd = kernel.K(Xnew, inducing_variable.Z) @ res_upd # [S, P, N, 1]
    samples = tf.linalg.adjoint( tf.squeeze(fX + f_upd, axis = 3) )
    return samples # [S, N, P]