import numpy as np
import tensorflow as tf
import gpflow
import invgp
import matplotlib.pyplot as plt
import numpy.random as rnd
from invgp.models import sample_SVGP
from gpflow.models import SVGP
from . import utils


np.random.seed(0)


def test_regression():
  # generate 200 datapoints
  X = np.random.uniform(0, 6, 200)[:, None]
  Y = np.sin(2 * X) + 0.1 * np.cos(7 * X) + np.random.randn(*X.shape) * 0.1
  train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
  train_dataset = train_dataset.shuffle(1024).batch(50)

  # initialize SVGP model
  nr_inducing_points = 50
  inducing_variables = X[rnd.permutation(len(X))[:nr_inducing_points], :]
  kernel = gpflow.kernels.SquaredExponential()
  likelihood = gpflow.likelihoods.Gaussian()

  SVGP_model = SVGP(kernel, likelihood,
                                inducing_variable=inducing_variables,
                                num_data=200)
  sample_SVGP_model = sample_SVGP.sample_SVGP(kernel, likelihood,
                                 inducing_variable=inducing_variables,
                                 num_data=200)
  matheron_sample_SVGP_model = sample_SVGP.sample_SVGP(kernel, likelihood,
                                 inducing_variable=inducing_variables,
                                 num_data=200,
                                 matheron_sampler=True)

  # train SVGP model
  train_iter = iter(train_dataset.repeat())
  training_loss = SVGP_model.training_loss_closure(train_iter, compile=True)
  optimizer = tf.keras.optimizers.Adam()
  @tf.function
  def optimization_step():
      optimizer.minimize(training_loss, SVGP_model.trainable_variables)
  for step in range(3500):
      optimization_step()
      minibatch_elbo = -training_loss().numpy()
      print('Step: %s, Mini batch elbo: %s' % (step, minibatch_elbo))

  # initialize sample SVGP models with fitted parameters from "regular" SVGP
  utils.initialize_with_trained_params(sample_SVGP_model, SVGP_model)
  utils.initialize_with_trained_params(matheron_sample_SVGP_model, SVGP_model)

  #compare elbos
  SVGP_model_elbo = SVGP_model.elbo((X, Y))

  sample_SVGP_model_elbos = [sample_SVGP_model.elbo((X, Y)) for _ in range(10)]
  expected_sample_elbo = np.mean([elbo.numpy() for elbo in sample_SVGP_model_elbos])
  np.testing.assert_allclose(SVGP_model_elbo, expected_sample_elbo, rtol=0.05, atol=0.0) 

  matheron_sample_SVGP_model_elbos = [matheron_sample_SVGP_model.elbo((X, Y)) for _ in range(10)]
  expected_matheron_sample_elbo = np.mean([elbo.numpy() for elbo in matheron_sample_SVGP_model_elbos])
  np.testing.assert_allclose(SVGP_model_elbo, expected_matheron_sample_elbo, rtol=0.05, atol=0.0)




