"""
Demo of traning a model with a kernel that is not calculated in closed-form.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import gpflow
import invgp
import invgp.kernels.orbits as orbits
from gpflow.utilities import set_trainable
from invgp.inducing_variables import StochasticConvolvedInducingPoints

stoch_k = invgp.kernels.StochasticInvariant(gpflow.kernels.SquaredExponential(lengthscales=2.0),
                                            orbits.GaussianNoiseOrbit(variance=0.2))
stoch_k.orbit.minibatch_size = 100
newlen = (stoch_k.basekern.lengthscales ** 2.0 + 2 * stoch_k.orbit.variance).numpy() ** 0.5
det_k = gpflow.kernels.SquaredExponential(lengthscales=newlen,
                                          variance=np.sqrt(stoch_k.basekern.lengthscales ** 2.0 / newlen ** 2.0))

# Generate data
pX = np.linspace(-4.0, 10.0, 200)[:, None]
X = np.random.uniform(0, 6, 200)[:, None]
Y = np.sin(2 * X) + 0.1 * np.cos(7 * X) + np.random.randn(*X.shape) * 0.1
Z = np.linspace(-0.5, 6.5, 20)[:, None]

#
# Train deterministic model
det_m = gpflow.models.SGPR((X, Y), det_k, inducing_variable=Z.copy())
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(tf.function(lambda: -det_m.elbo()),
                        det_m.trainable_variables,
                        options=dict(maxiter=1000))
m, v = det_m.predict_y(pX)

_, ax = plt.subplots()
ax.plot(X, Y, 'x')
ax.plot(pX, m)
ax.plot(pX, m + 2 * v ** 0.5, color='C1')
ax.plot(pX, m - 2 * v ** 0.5, color='C1')

#
# Train stochastic model
stoch_m = gpflow.models.SVGP(stoch_k, gpflow.likelihoods.Gaussian(), StochasticConvolvedInducingPoints(Z))
stoch_m.likelihood.variance.assign(0.1)
set_trainable(stoch_m.inducing_variable, False)


@tf.function()
def optimization_step(optimizer, model: gpflow.models.SVGP, batch):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        objective = - model.elbo(batch)
        grads = tape.gradient(objective, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return objective


def run_adam(model, iterations):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    adam = tf.optimizers.Adam()
    for step in range(iterations):
        elbo = - optimization_step(adam, model, (X, Y))
        if step % 10 == 0:
            logf.append(elbo.numpy())
            print(f"{step}, {elbo.numpy()}", end="\r")
    return logf


opt = tf.optimizers.Adam(learning_rate=1e-3)
hist = run_adam(stoch_m, 10000)

m, v = stoch_m.predict_y(pX)

_, ax = plt.subplots()
ax.plot(X, Y, 'x')
ax.plot(pX, m)
ax.plot(pX, m + 2 * v ** 0.5, color='C1')
ax.plot(pX, m - 2 * v ** 0.5, color='C1')

plt.show()
