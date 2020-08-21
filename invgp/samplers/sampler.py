import tensorflow as tf
from ..inducing_variables import ConvolvedInducingPoints
from ..kernels import Invariant
from gpflow.utilities import Dispatcher
from gpflow.kernels import Kernel

sample_matheron = Dispatcher("sample_matheron")

# the same could be achieved by dispatching on the sampler?
@sample_matheron.register(object,object, object, Kernel)
def _sample_matheron(self, Xnew, inducing_variable, kernel):
    return self.sampler(Xnew) # [S_g, N, P]


@sample_matheron.register(object, object, ConvolvedInducingPoints, Invariant)
def _sample_matheron(self, Xnew, inducing_variable, kernel):
    X_o = kernel.orbit(Xnew)
    Xnew = tf.reshape(X_o, [X_o.shape[0]*X_o.shape[1], X_o.shape[2]]) # [NS_o, D]
    samples = sample_matheron(self, Xnew,  inducing_variable.Z, kernel.basekern) # [S_g, NS_o, P]
    samples = tf.reshape(samples, [self.num_samples, X_o.shape[0], X_o.shape[1], samples.shape[2]]) # [S, N, C, P]
    samples = tf.reduce_mean(samples, axis = 2) # [S, N, P] 
    return samples
