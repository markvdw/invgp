import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import gpflow
import invgp

X = np.array([[1.0, 0.0], [1.0, 0.03], [1.0, 1.0]])
Y = np.array([[1.0], [0.9], [0.0]])

m = gpflow.models.SVGP(invgp.kernels.SwitchDimsInvariant(gpflow.kernels.SquaredExponential()),
                       gpflow.likelihoods.Gaussian(),
                       gpflow.inducing_variables.InducingPoints(X))

opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(tf.function(lambda: -m.elbo((X, Y))),
                        m.trainable_variables,
                        options=dict(maxiter=1000))

x = np.linspace(-2, 2, 50)
X1, X2 = np.meshgrid(x, x, indexing='ij')
Xt = np.c_[X1.flatten(), X2.flatten()]
pred = m.predict_f(Xt.astype(gpflow.config.default_float()))[0].numpy()

plt.imshow(pred.reshape(len(x), len(x)))
plt.show()
