import numpy as np
import tensorflow as tf
import gpflow
import invgp
import matplotlib.pyplot as plt
import numpy.random as rnd
from invgp.inducing_variables.invariant_convolution_domain import StochasticConvolvedInducingPoints
from invgp.models import sample_SVGP
from gpflow.models import SVGP
import utils

np.random.seed(0)


# generate 200 datapoints
X = np.random.uniform(-3, 3, 400)[:, None]
X = np.reshape(X, [200, 2]) # 2-dimensional input
M = np.ones([2, 1]) * 2
#Y = np.matmul(X, M) # y = 2x_1 + 2x_2
Y = np.sqrt(X[:, 0]**2 + X[:, 1]**2)[..., np.newaxis]
train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
train_dataset = train_dataset.shuffle(1024).batch(50)

# initialize SVGP model
nr_inducing_points = 50
inducing_variables = X[rnd.permutation(len(X))[:nr_inducing_points], :]
inducing_variables = StochasticConvolvedInducingPoints(inducing_variables)
basekernel = gpflow.kernels.SquaredExponential()
orbit = invgp.kernels.orbits.SwitchXY()
kernel = invgp.kernels.StochasticInvariant(
                 basekern=basekernel,
                 orbit=orbit)
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
for step in range(10000):
   optimization_step()
   minibatch_elbo = -training_loss().numpy()
   print('Step: %s, Mini batch elbo: %s' % (step, minibatch_elbo))

# initialize sample SVGP model with fitted parameters from SVGP
utils.initialize_with_trained_params(sample_SVGP_model, SVGP_model)
utils.initialize_with_trained_params(matheron_sample_SVGP_model, SVGP_model)
gpflow.utilities.print_summary(SVGP_model)
gpflow.utilities.print_summary(sample_SVGP_model)
gpflow.utilities.print_summary(matheron_sample_SVGP_model)

fig, ax = plt.subplots(1, 3)
x1list = np.linspace(-3.0, 3.0, 100)
x2list = np.linspace(-3.0, 3.0, 100)
X1, X2 = np.meshgrid(x1list, x2list)
ax[0].set_title('True function')
ax[0].set_aspect('equal', 'box')

# plot the true data generating process
true_Z = np.sqrt(X1**2 + X2**2)
cp = ax[0].contourf(X1, X2, true_Z)

# plot the SVGP mean prediction
positions = np.vstack([X1.ravel(), X2.ravel()])
mean, var = SVGP_model.predict_f(positions.T)
cp = ax[1].contourf(X1, X2, np.reshape(mean.numpy().T, X1.shape))
ax[1].set_title('Posterior mean')
ax[1].set_aspect('equal', 'box')

# plot a matheron sample_SVGP sample
x1list_coarse = np.linspace(-3.0, 3.0, 10)
x2list_coarse = np.linspace(-3.0, 3.0, 10)
X1_coarse, X2_coarse = np.meshgrid(x1list_coarse, x2list_coarse)
positions_coarse = np.vstack([X1_coarse.ravel(), X2_coarse.ravel()])
samples = matheron_sample_SVGP_model.predict_f_samples(positions_coarse.T, num_samples=1)
cp = ax[2].contourf(X1_coarse, X2_coarse, np.reshape(samples.numpy().T, X1_coarse.shape))
ax[2].set_title('f sample from Matheron model')
ax[2].set_aspect('equal', 'box')

fig.colorbar(cp) # Add a colorbar to a plot
plt.show()

#compare elbos
SVGP_model_elbo = SVGP_model.elbo((X, Y))
print('SVGP model elbo is:', SVGP_model_elbo.numpy())

sample_SVGP_model_elbos = [sample_SVGP_model.elbo((X, Y)) for _ in range(10)]
expected_sample_elbo = np.mean([elbo.numpy() for elbo in sample_SVGP_model_elbos])
print('sample_SVGP model elbos:', [elbo.numpy() for elbo in sample_SVGP_model_elbos])
print('Expectation of the sample ELBO estimator:', expected_sample_elbo)
np.testing.assert_allclose(SVGP_model_elbo, expected_sample_elbo, rtol=1.0, atol=0.0) # the tolerance is picked somewhat randomly

matheron_sample_SVGP_model_elbos = [matheron_sample_SVGP_model.elbo((X, Y)) for _ in range(10)]
expected_matheron_sample_elbo = np.mean([elbo.numpy() for elbo in matheron_sample_SVGP_model_elbos])
print('Matheron sample_SVGP model elbos:', [elbo.numpy() for elbo in matheron_sample_SVGP_model_elbos])
print('Expectation of the Matheron sample ELBO estimator:', expected_matheron_sample_elbo)
np.testing.assert_allclose(SVGP_model_elbo, expected_matheron_sample_elbo, rtol=1.0, atol=0.0) # the tolerance is picked somewhat randomly




