import tensorflow as tf
from gpflow.conditionals import sample_conditional
from gpflow.models import SVGP
from gpflow.models.model import InputData, RegressionData

from invgp.samplers import sample_matheron

from typing import Tuple


class SampleSVGP(SVGP):
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
        num_samples: int = 5,
        matheron_sampler: bool = False,
        num_basis: int = 1024,
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
        super().__init__(
            kernel,
            likelihood,
            inducing_variable,
            num_latent_gps=num_latent_gps,
            whiten=whiten,
            num_data=num_data,
            mean_function=mean_function,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
            q_diag=q_diag,
        )

        self.matheron_sampler = matheron_sampler
        self._sampler = None
        self.num_samples = num_samples
        self.num_basis = num_basis

    def elbo(self, data: RegressionData):
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model using samples from f.

        """
        X, Y = data  # X is [N x D]
        kl = self.prior_kl()

        f_samples = self.predict_f_samples(
            X,
            num_samples=self.num_samples,
            full_cov=True,
            full_output_cov=False,
            matheron_sampler=self.matheron_sampler)  # [S_f, N, P]

        Y_ = tf.tile(Y, [self.num_samples, 1]) # [S_f*N, P]
        f_samples = tf.reshape(f_samples, [-1, f_samples.shape[2]])  # [S_f*N, P]
        likelihoods = self.likelihood.log_prob(f_samples, Y_)  # [S_f*N]
        likelihoods = tf.reshape(likelihoods, [self.num_samples, -1])  # [S_f, N]
        exp_likelihood = tf.reduce_mean(likelihoods, axis=0)  # N

        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(exp_likelihood) * scale - kl

    def predict_f_samples(
        self,
        Xnew: InputData,
        num_samples: int = None,
        full_cov: bool = True,
        full_output_cov: bool = False,
        matheron_sampler: bool = False,
    ) -> tf.Tensor:
        if matheron_sampler:
            samples = sample_matheron(
                Xnew,
                self.inducing_variable,
                self.kernel,
                self.q_mu,
                self.q_sqrt,
                white=self.whiten,
                num_samples=num_samples,
            )
        else:
            samples, _, _ = sample_conditional(
                Xnew,
                self.inducing_variable,
                self.kernel,
                self.q_mu,
                full_cov=full_cov,
                full_output_cov=full_output_cov,
                q_sqrt=self.q_sqrt,
                white=self.whiten,
                num_samples=num_samples,
            )


        return samples  # [..., (S), N, P]

    def predict_y(
        self,
        Xnew: InputData,
        num_samples: int = None,
        full_cov: bool = True,
        full_output_cov: bool = False,
        matheron_sampler: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        f_samples = self.predict_f_samples(Xnew, full_cov = True, full_output_cov = False, num_samples = self.num_samples, matheron_sampler = self.matheron_sampler)

        f_mean = tf.reduce_mean(f_samples, axis = 0)
        f_var = tf.reduce_mean(f_samples**2, axis = 0) - f_mean**2

        return self.likelihood.predict_mean_and_var(f_mean, f_var)

        
