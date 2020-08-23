import tensorflow as tf
import numpy as np
from invgp.inducing_variables import ConvolvedInducingPoints
from invgp.kernels import Invariant
from gpflow.utilities import Dispatcher
from gpflow.kernels import Kernel
from gpflow.inducing_variables import InducingPoints
from gpflow import kernels

sample_matheron = Dispatcher("sample_matheron")

@sample_matheron.register(object, object, Invariant, object, object)
def _sample_matheron(Xnew, inducing_variable, kernel, q_mu, q_sqrt, num_samples = 1, num_basis = 1024):
    X_o = kernel.orbit(Xnew)
    print('hello')
    Xnew = tf.reshape(X_o, [X_o.shape[0]*X_o.shape[1], X_o.shape[2]]) # [NS_o, D]
    samples = sample_matheron(Xnew,  inducing_variable, kernel.basekern, q_mu, q_sqrt, num_samples=num_samples, num_basis=num_basis) 
    samples = tf.reshape(samples, [num_samples, X_o.shape[0], X_o.shape[1], samples.shape[2]]) # [S, N, C, P]
    samples = tf.reduce_mean(samples, axis = 2) # [S, N, P] 
    return samples

@sample_matheron.register(object, object, Kernel, object, object)
def __sample_matheron(Xnew, inducing_variable, kernel, q_mu, q_sqrt, num_samples = 1, num_basis = 1024):
    
    print(kernel)
    if not isinstance(kernel, kernels.SquaredExponential):
        raise NotImplementedError
    
    # Draw from prior
    bias = tf.random.uniform(shape = [num_basis], maxval=2*np.pi, dtype = Xnew.dtype)
    spectrum = tf.random.normal(shape = [num_basis, Xnew.shape[1]], dtype = Xnew.dtype)
    _Xnew = tf.divide(Xnew,kernel.lengthscales)
    _Z = tf.divide(inducing_variable.Z, kernel.lengthscales)
    
    w = tf.random.normal(shape = [num_samples, q_mu.shape[1], num_basis, 1], dtype = Xnew.dtype) # [S, P, num_basis, 1]
    print('Xshape:', _Xnew.shape)
    print('w shape:', w.shape)
    print('spectrum:', spectrum.shape)
    print('bias:', bias.shape)
    fX = tf.sqrt(kernel.variance * 2 / num_basis) * tf.cos(tf.matmul(_Xnew, spectrum, transpose_b = True) + bias) # [N, num_basis]
    fX = tf.tile(fX[None,None,:,:], [num_samples, q_mu.shape[1], 1,1] ) @ w  # [S, P, N, 1]
    fZ = tf.sqrt(kernel.variance * 2 / num_basis) * tf.cos(tf.matmul(_Z, spectrum, transpose_b = True) + bias) # [M, num_basis]
    fZ = tf.tile(fZ[None,None,:,:], [num_samples, q_mu.shape[1], 1,1] ) @ w  # [S, P, M, 1]
    
    # Update
    v = tf.random.normal(shape = [num_samples, q_mu.shape[1], q_mu.shape[0], 1], dtype = Xnew.dtype) # [S, P, M, 1]
    print(q_mu.shape)
    _u = tf.tile(tf.linalg.adjoint(q_mu)[None, :, :, None], [num_samples, 1,1,1]) +  tf.matmul(tf.tile(q_sqrt[None,:,:,:], [num_samples,1,1,1]), v)# [S, P, M, 1]
    
    Luu = tf.linalg.cholesky(kernel.K(inducing_variable.Z)) # [M, M]
    print('Luushape:', Luu.shape)
    res_upd = tf.linalg.cholesky_solve(tf.tile(Luu[None, None, :, :], [num_samples, q_mu.shape[1],1,1]) , _u - fZ) # [S, P, M, 1]
    print('res_upd shape:', res_upd.shape)
    print('K shape:', kernel.K(Xnew, inducing_variable.Z)) #[N, M]
    f_upd = kernel.K(Xnew, inducing_variable.Z) @ res_upd # [S, P, N, 1]
    samples = tf.linalg.adjoint( tf.squeeze(fX + f_upd, axis = 3) )
    return samples # [S, N, P]