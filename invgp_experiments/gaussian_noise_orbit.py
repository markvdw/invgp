import matplotlib.pyplot as plt
import numpy as np

import gpflow
import invgp
import invgp.kernels.orbits as orbits

k = invgp.kernels.StochasticInvariant(gpflow.kernels.SquaredExponential(), orbits.GaussianNoiseOrbit(variance=0.02))
k.orbit.minibatch_size = 100
newlen = (k.basekern.lengthscales + 2 * k.orbit.variance).numpy() ** 0.5
comp_k = gpflow.kernels.SquaredExponential(lengthscales=newlen,
                                           variance=np.sqrt(k.basekern.lengthscales / newlen ** 2.0))

X = np.linspace(-5, 5, 100)[:, None]
X2 = np.array([[0.0]])

stoch_kernfunc = k.K(X, X2)
base_kernfunc = k.basekern.K(X, X2)
comp_kernfunc = comp_k.K(X, X2)

plt.plot(stoch_kernfunc)
plt.plot(base_kernfunc)
plt.plot(comp_kernfunc)
plt.show()
