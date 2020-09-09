import tensorflow as tf

from gpflow.likelihoods.base import MonteCarloLikelihood

class SampleSoftmax(MonteCarloLikelihood):
    """
    A remake of the softmax likelihood, compatible with SampleSVGP
    """

    def __init__(self, num_classes, **kwargs):
        super().__init__(latent_dim=num_classes, observation_dim=None, **kwargs)
        self.num_classes = self.latent_dim

    def _log_prob(self, F, Y):
        #print(Y.shape)
        #Y = tf.tile(Y[None,:,:], [F.shape[0],1,1])
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=F, labels=Y[:,:, 0])

    def _conditional_mean(self, F):
        raise NotImplementedError
        #return tf.nn.softmax(F)

    def _conditional_variance(self, F):
        raise NotImplementedError
        #p = self.conditional_mean(F)
        #return p - p ** 2
