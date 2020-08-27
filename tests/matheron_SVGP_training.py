import numpy as np
import tensorflow as tf
import gpflow
import invgp
import matplotlib.pyplot as plt
import numpy.random as rnd
from invgp.models import SampleSVGP
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
matheron_sample_SVGP_model = SampleSVGP.SampleSVGP(kernel, likelihood,
                                                   inducing_variable=inducing_variables,
                                                   num_data=200,
                                                   num_samples=100,
                                                   matheron_sampler=True)
SVGP_model = SVGP(kernel, likelihood,
                              inducing_variable=inducing_variables,
                              num_data=200)

print('untrained model elbos:', SVGP_model.elbo((X, Y)), matheron_sample_SVGP_model.elbo((X, Y)) )
gpflow.utilities.print_summary(SVGP_model)
gpflow.utilities.print_summary(matheron_sample_SVGP_model)

# train SVGP model
train_iter = iter(train_dataset.repeat())
training_loss = matheron_sample_SVGP_model.training_loss_closure(train_iter, compile=True)
optimizer = tf.keras.optimizers.Adam()
@tf.function
def optimization_step():
    optimizer.minimize(training_loss, matheron_sample_SVGP_model.trainable_variables)
for step in range(3500):
    optimization_step()
    minibatch_elbo = -training_loss().numpy()
    print('Step: %s, Mini batch elbo: %s' % (step, minibatch_elbo))

gpflow.utilities.print_summary(matheron_sample_SVGP_model)

# visualize data and fitted model
# test points
xx = np.linspace(-0.1, 6.1, 100).reshape(100, 1)
# predict mean and variance of latent GP at test points
mean, var = matheron_sample_SVGP_model.predict_f(xx)
# generate 10 samples from posterior
tf.random.set_seed(1)  # for reproducibility
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
plt.plot(xx, matheron_SVGP_samples[:, :, 0].numpy().T, "C0", linewidth=0.5, c='darkred', label='Matheron SVGP')
plt.legend()
_ = plt.xlim(-0.1, 6.1)
plt.show()

