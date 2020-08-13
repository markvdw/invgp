from gpflow.models import SVGP
from gpflow.models.model import GPModel, InputData, RegressionData, MeanAndVariance
import numpy as np
import tensorflow as tf


class sample_SVGP(SVGP):
    def __init__(
        self,
        kernel,
        likelihood,
        inducing_variable,
        *,
        mean_function=None,
        num_latent_gps: int = 1,
        q_diag: bool = False,
        q_mu=None,
        q_sqrt=None,
        whiten: bool = True,
        num_data=None,
        S_a_S_g: int = 5,
    ):
        """
        - kernel, likelihood, inducing_variables, mean_function are appropriate
          GPflow objects
        - num_latent_gps is the number of latent processes to use, defaults to 1
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # init the super class, accept args
        super().__init__(kernel, likelihood, inducing_variable, num_latent_gps=num_latent_gps)
        self.S_a_S_g = S_a_S_g


    def elbo(self, data: RegressionData): 
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model using samples from f. 

        """
        X, Y = data # X is [N x D]
        # initilize some dimensions
        S_o = X.shape[1]
        N = X.shape[0]
        D = X.shape[1]
        P = self.num_latent_gps

        kl = self.prior_kl()
        print('X shape is', X.shape)
        f_samples = self.predict_f_samples(X, num_samples=self.S_a_S_g, full_cov=False, full_output_cov=False) # now: [S_a_S_g, N, P] (later: [S_a, S_g, N, P] ?)
        print('f_samples shape is', f_samples.shape) 

        # expand Y to the right dimensions
        Y = tf.expand_dims(Y, 0) # (1, 50, 10)
        Y = tf.tile(Y, [self.S_a_S_g, 1, 1])
        # compute likelihoods
        likelihoods = self.likelihood.log_prob(f_samples, Y)
        # compute expectations over g and X_a
        exp_likelihood = tf.reduce_mean(likelihoods, axis=0) # N

        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(exp_likelihood) * scale - kl


#     def predict_f_samples(                        # TODO: override from GPmodels
#        self,
#        Xnew: InputData,
#        num_samples: Optional[int] = None,
#        full_cov: bool = True,
#        full_output_cov: bool = False,
#    ) -> tf.Tensor:
        """
        Produce samples from the posterior latent function(s) at the input points.

        :param Xnew: InputData
            Input locations at which to draw samples, shape [..., N, D]
            where N is the number of rows and D is the input dimension of each point.
        :param num_samples:
            Number of samples to draw.
            If `None`, a single sample is drawn and the return shape is [..., N, P],
            for any positive integer the return shape contains an extra batch
            dimension, [..., S, N, P], with S = num_samples and P is the number of outputs.
        :param full_cov:
            If True, draw correlated samples over the inputs. Computes the Cholesky over the
            dense covariance matrix of size [num_data, num_data].
            If False, draw samples that are uncorrelated over the inputs.
        :param full_output_cov:
            If True, draw correlated samples over the outputs.
            If False, draw samples that are uncorrelated over the outputs.

        Currently, the method does not support `full_output_cov=True` and `full_cov=True`.
        """
        # if full_cov and full_output_cov:
        #     raise NotImplementedError(
        #         "The combination of both `full_cov` and `full_output_cov` is not supported."
        #     )

        # # check below for shape info
        # mean, cov = self.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        # if full_cov:
        #     # mean: [..., N, P]
        #     # cov: [..., P, N, N]
        #     mean_for_sample = tf.linalg.adjoint(mean)  # [..., P, N]
        #     samples = sample_mvn(
        #         mean_for_sample, cov, full_cov, num_samples=num_samples
        #     )  # [..., (S), P, N]
        #     samples = tf.linalg.adjoint(samples)  # [..., (S), N, P]
        # else:
        #     # mean: [..., N, P]
        #     # cov: [..., N, P] or [..., N, P, P]
        #     samples = sample_mvn(
        #         mean, cov, full_output_cov, num_samples=num_samples
        #     )  # [..., (S), N, P]
        # return samples  # [..., (S), N, P]