from invgp.kernels import orbits
from invgp_experiments.datasets import load_mnist
import matplotlib.pyplot as plt
import tensorflow as tf

(X, Y), _ = load_mnist()

# parametrization: [angle_deg, sx, sy, tx, ty], [0., 1., 1., 0., 0.]

orbit = orbits.ColorTransform(minibatch_size=10, log_lims_contrast=[-1., 1.], log_lims_brightness=[-1., 1.])
Xo = orbit(X[:2, :])

fig, ax = plt.subplots(11, 2)
ax[0, 0].imshow(tf.reshape(X[0], (28, 28)), cmap='gray', vmin=0, vmax=1)
ax[0, 1].imshow(tf.reshape(X[1], (28, 28)), cmap='gray', vmin=0, vmax=1)
for i in range(1, 11):
	ax[i, 0].imshow(tf.reshape(Xo[0, i-1, :], (28, 28)), cmap='gray', vmin=0, vmax=1)
	ax[i, 1].imshow(tf.reshape(Xo[1, i-1, :], (28, 28)), cmap='gray', vmin=0, vmax=1)
plt.show()