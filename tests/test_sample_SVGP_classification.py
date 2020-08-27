# import numpy as np
# import tensorflow as tf
# import gpflow
# import invgp
# from invgp_experiments import datasets
# import matplotlib.pyplot as plt
# import numpy.random as rnd
# from invgp.inducing_variables.invariant_convolution_domain import StochasticConvolvedInducingPoints
# from invgp.models import sample_SVGP
# from gpflow.models import SVGP
# from . import utils
# #from deepkernelinv.kernels import DeepKernel
# #from deepkernelinv.models.inducing_variables import ConvolvedKernelSpaceInducingPoints
#
# np.random.seed(0)
#
# def test_classification():
#   # Generate dataset: a small subset of MNIST
#   (X, Y), _ = datasets.load_mnist()
#   X = X[:1000, :]
#   # X = X.reshape(500, 28, 28, 1) # uncomment for DeepKernel
#   Y = Y[:1000, 0]
#   Y = tf.one_hot(tf.cast(Y, tf.uint8), 10)
#   Y = tf.cast(Y, tf.float64)
#   print('data shapes are', X.shape, Y.shape)
#
#   # initialize models
#   nr_inducing_points = 100
#   inducing_variables = X[rnd.permutation(len(X))[:nr_inducing_points], :]
#   kernel = gpflow.kernels.SquaredExponential()
#   # TODO: test for DeepKernel and invariant DeepKernel models
#   # kernel = DeepKernel(
#   #         image_shape=(28, 28, 1),
#   #         filters_l1=20,
#   #         filters_l2=50,
#   #         hidden_units=500,
#   #         output_units=50,
#   #         basekern=kernel,
#   #         batch_size=50)
#   # inducing_variables = ConvolvedKernelSpaceInducingPoints(kernel.cnn(inducing_variables))
#   likelihood = gpflow.likelihoods.Gaussian()
#
#   sample_SVGP_model = sample_SVGP.sample_SVGP(kernel, likelihood,
#                                  inducing_variable=inducing_variables,
#                                  num_data=1000,
#                                  num_latent_gps=10)
#   matheron_sample_SVGP_model = sample_SVGP.sample_SVGP(kernel, likelihood,
#                                  inducing_variable=inducing_variables,
#                                  num_data=1000,
#                                  num_latent_gps=10,
#                                  matheron_sampler=True)
#   SVGP_model = SVGP(kernel, likelihood,
#                                  inducing_variable=inducing_variables,
#                                  num_data=1000,
#                                  num_latent_gps=10)
#
#
#   # train SVGP model
#   train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
#   train_dataset = train_dataset.shuffle(1024).batch(100)
#   train_iter = iter(train_dataset.repeat())
#   training_loss = SVGP_model.training_loss_closure(train_iter, compile=True)
#   optimizer = tf.keras.optimizers.Adam()
#   @tf.function
#   def optimization_step():
#       optimizer.minimize(training_loss, SVGP_model.trainable_variables)
#   for step in range(10000):
#       optimization_step()
#       minibatch_elbo = -training_loss().numpy()
#       print('Step: %s, Mini batch elbo: %s' % (step, minibatch_elbo))
#
#   # initialize sample SVGP models with fitted parameters from "regular" SVGP
#   utils.initialize_with_trained_params(sample_SVGP_model, SVGP_model)
#   utils.initialize_with_trained_params(matheron_sample_SVGP_model, SVGP_model)
#   gpflow.utilities.print_summary(SVGP_model)
#   gpflow.utilities.print_summary(sample_SVGP_model)
#   gpflow.utilities.print_summary(matheron_sample_SVGP_model)
#
#   #compare elbos
#   SVGP_model_elbo = SVGP_model.elbo((X, Y))
#   print('SVGP model elbo is:', SVGP_model_elbo.numpy())
#
#   sample_SVGP_model_elbos = [sample_SVGP_model.elbo((X, Y)) for _ in range(10)]
#   expected_sample_elbo = np.mean([elbo.numpy() for elbo in sample_SVGP_model_elbos])
#   print('sample_SVGP model elbos:', [elbo.numpy() for elbo in sample_SVGP_model_elbos])
#   print('Expectation of the sample ELBO estimator:', expected_sample_elbo)
#   np.testing.assert_allclose(SVGP_model_elbo, expected_sample_elbo, rtol=0.05, atol=0.0) # check that values are within 5% of each other
#
#   matheron_sample_SVGP_model_elbos = [matheron_sample_SVGP_model.elbo((X, Y)) for _ in range(10)]
#   expected_matheron_sample_elbo = np.mean([elbo.numpy() for elbo in matheron_sample_SVGP_model_elbos])
#   print('Matheron sample_SVGP model elbos:', [elbo.numpy() for elbo in matheron_sample_SVGP_model_elbos])
#   print('Expectation of the Matheron sample ELBO estimator:', expected_matheron_sample_elbo)
#   np.testing.assert_allclose(SVGP_model_elbo, expected_matheron_sample_elbo, rtol=0.05, atol=0.0)
#
#
#
#
