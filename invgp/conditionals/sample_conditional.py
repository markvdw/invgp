from gpflow.conditionals.dispatch import sample_conditional
from gpflow.inducing_variables import InducingPoints
from ..inducing_variables import ConvolvedInducingPoints
from ..kernels import Invariant
import tensorflow as tf

# @sample_conditional.register(object, InducingPoints, InvDeepKernel, object)
# @sample_conditional.register(object, InducingPoints, DeepKernel, object)
# seems hacky to dispatch to these just in order to throw an error?
@sample_conditional.register(object, InducingPoints, Invariant, object)
def _sample_conditional(Xnew,
            inducing_variable,
            kernel,
            q_mu,
            *,
            full_cov=False,
            full_output_cov=False,
            q_sqrt=None,
            white=False,
            num_samples=None):
    print('Running invariant kernel sample_conditional.')

    X_o = kernel.orbit(Xnew) # [N, C, D]
    Xnew = tf.reshape(X_o, [-1, X_o.shape[2]]) # [NC, D]
    # Xnew = tf.reshape(X_o, [X_o.shape[0]*X_o.shape[1], X_o.shape[2]]) # [NC, D]
    samples, mean, cov = sample_conditional(
        Xnew,
        inducing_variable.Z,
        kernel.basekern,
        q_mu,
        q_sqrt=q_sqrt,
        full_cov=full_cov,
        white=white,
        full_output_cov=full_output_cov,
        num_samples=num_samples)
    samples = tf.reshape(samples, [num_samples, -1, X_o.shape[1], samples.shape[2]]) # [S, N, C, P]
    #samples = tf.reshape(samples, [num_samples, X_o.shape[0], X_o.shape[1], samples.shape[2]]) # [S, N, C, P]
    samples = tf.reduce_mean(samples, axis = 2) # [S, N, P]
    return samples, mean, cov