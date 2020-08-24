import numpy as np
import tensorflow as tf
import gpflow
import invgp
import matplotlib.pyplot as plt
import numpy.random as rnd
from invgp.models import sample_SVGP
from gpflow.models import SVGP
import utils

np.random.seed(0)

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
print('\n untrained model elbos:', SVGP_model.elbo((X, Y)).numpy(),
  sample_SVGP_model.elbo((X, Y)).numpy(), matheron_sample_SVGP_model.elbo((X, Y)).numpy())

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
gpflow.utilities.print_summary(SVGP_model)
gpflow.utilities.print_summary(sample_SVGP_model)
gpflow.utilities.print_summary(matheron_sample_SVGP_model)

# visualize data and fitted model
# test points
xx = np.linspace(-0.1, 6.1, 100).reshape(100, 1)
# predict mean and variance of latent GP at test points
mean, var = SVGP_model.predict_f(xx)
# generate 10 samples from posterior
tf.random.set_seed(1)  # for reproducibility
SVGP_samples = SVGP_model.predict_f_samples(xx, 10)  # shape (10, 100, 1)
sample_SVGP_samples = sample_SVGP_model.predict_f_samples(xx, 10)  # shape (10, 100, 1)
matheron_SVGP_samples = matheron_sample_SVGP_model.predict_f_samples(xx, 10)

## plot
plt.figure(figsize=(12, 6))
plt.plot(X, Y, "kx", mew=2)
plt.plot(xx, mean, "C0", lw=2)
plt.fill_between(
    xx[:, 0],
    mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
    mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
    color="C0",
    alpha=0.2,)
plt.plot(xx, SVGP_samples[:, :, 0].numpy().T, "C0", linewidth=0.5, label='SVGP')
plt.plot(xx, sample_SVGP_samples[:, :, 0].numpy().T, "C0", linewidth=0.5, c='darkgreen', label='Sample SVGP')
plt.plot(xx, matheron_SVGP_samples[:, :, 0].numpy().T, "C0", linewidth=0.5, c='darkred', label='Matheron SVGP')
plt.legend()
_ = plt.xlim(-0.1, 6.1)
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




