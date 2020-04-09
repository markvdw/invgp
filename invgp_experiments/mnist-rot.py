import numpy as np
import tensorflow as tf

import gpflow
import invgp
from gpflow.utilities import set_trainable
from datasets import load_mnist

np.random.seed(1)

# Generate dataset
(mnist_X_23, mnist_Y_23), _ = load_mnist(digits=[2, 3])
random_subset = np.random.permutation(len(mnist_X_23))[:97]
mnist_X_23 = mnist_X_23[random_subset, :]
mnist_Y_23 = mnist_Y_23[random_subset, :]

gen_orbit = invgp.kernels.orbits.ImageRotQuant(3, interpolation_method="BILINEAR")
orbit_X = gen_orbit(mnist_X_23).numpy().reshape(-1, mnist_X_23.shape[1])
orbit_Y = np.repeat(mnist_Y_23[:, None, :], gen_orbit.orbit_size, 1).reshape(-1, 1)
random_perm = np.random.permutation(len(orbit_X))
X = orbit_X[random_perm[:500], :]
Y = (orbit_Y[random_perm[:500], :] == 2.0).astype('float') * 2.0 - 1.0
Xt = orbit_X[random_perm[500:1000], :]
Yt = (orbit_Y[random_perm[500:1000], :] == 2.0).astype('float') * 2.0 - 1.0

#
# Squared exponential model
sqexp_m = gpflow.models.GPR((X, Y), gpflow.kernels.SquaredExponential())
sqexp_m.kernel.lengthscales.assign(50)
set_trainable(sqexp_m.likelihood, False)
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(tf.function(lambda: -sqexp_m.elbo()),
                        sqexp_m.trainable_variables, options=dict(maxiter=1000, disp=True))
set_trainable(sqexp_m.likelihood, True)
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(tf.function(lambda: -sqexp_m.elbo()),
                        sqexp_m.trainable_variables, options=dict(maxiter=1000, disp=True))

sqexp_err = (1.0 - ((sqexp_m.predict_f(Xt)[0] > 0.0) == (Yt > 0.0)).numpy().mean()) * 100
gpflow.utilities.print_summary(sqexp_m)

#
# Rot90 model
rot90_k = invgp.kernels.Invariant(gpflow.kernels.SquaredExponential(),
                                  invgp.kernels.orbits.ImageRot90())
rot90_m = gpflow.models.GPR((X, Y), rot90_k)
rot90_m.kernel.basekern.lengthscales.assign(50)
set_trainable(rot90_m.likelihood, False)
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(tf.function(lambda: -rot90_m.elbo()),
                        rot90_m.trainable_variables, options=dict(maxiter=1000, disp=True))
set_trainable(rot90_m.likelihood, True)
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(tf.function(lambda: -rot90_m.elbo()),
                        rot90_m.trainable_variables, options=dict(maxiter=1000, disp=True))

rot90_err = (1.0 - ((rot90_m.predict_f(Xt)[0] > 0.0) == (Yt > 0.0)).numpy().mean()) * 100

#
# RotQuant model
rotq_k = invgp.kernels.Invariant(gpflow.kernels.SquaredExponential(),
                                 invgp.kernels.orbits.ImageRotQuant())
rotq_m = gpflow.models.GPR((X, Y), rotq_k)
rotq_m.kernel.basekern.lengthscales.assign(50)
set_trainable(rotq_m.likelihood, False)
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(tf.function(lambda: -rotq_m.elbo()),
                        rotq_m.trainable_variables, options=dict(maxiter=1000, disp=True))
set_trainable(rotq_m.likelihood, True)
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(tf.function(lambda: -rotq_m.elbo()),
                        rotq_m.trainable_variables, options=dict(maxiter=1000, disp=True))

rotq_err = (1.0 - ((rotq_m.predict_f(Xt)[0] > 0.0) == (Yt > 0.0)).numpy().mean()) * 100

print(f"sqexp_err: {sqexp_err:.2f}")
print(f"rot90_err: {rot90_err:.2f}")
print(f"rotq_err: {rotq_err:.2f}")
