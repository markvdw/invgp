from invgp.kernels import orbits
import matplotlib.pyplot as plt
import tensorflow as tf
from deepkernelinv.utils.load_datasets import load
import tensorflow_datasets as tfds
import numpy as np

#(X, Y), _ = load_mnist()

class args:
    dataset = "MNIST"
    image_shape = (32, 32, 3) if dataset == "CIFAR10" else (28, 28)
    subset_size = None

train, test = load(args)
X, _ = list(zip(*[a for a in tfds.as_numpy(train.take(7))]))
X = np.stack(X).astype('float64')

orbit = orbits.ColorTransform(minibatch_size=10, log_lims_contrast=[-2., 2.], log_lims_brightness=[-2., 2.])
Xo = orbit(X[:2, :])

if args.dataset == "CIFAR10":
	fig, ax = plt.subplots(11, 2)
	ax[0, 0].imshow(tf.reshape(X[0], args.image_shape))
	ax[0, 1].imshow(tf.reshape(X[1], args.image_shape))
	for i in range(1, 11):
		ax[i, 0].imshow(tf.reshape(Xo[0, i-1, :], args.image_shape))
		ax[i, 1].imshow(tf.reshape(Xo[1, i-1, :], args.image_shape))
	plt.show()


if args.dataset == "MNIST":
	fig, ax = plt.subplots(11, 2)
	ax[0, 0].imshow(tf.reshape(X[0], (28, 28)), cmap='gray', vmin=0, vmax=1)
	ax[0, 1].imshow(tf.reshape(X[1], (28, 28)), cmap='gray', vmin=0, vmax=1)
	for i in range(1, 11):
		ax[i, 0].imshow(tf.reshape(Xo[0, i-1, :], (28, 28)), cmap='gray', vmin=0, vmax=1)
		ax[i, 1].imshow(tf.reshape(Xo[1, i-1, :], (28, 28)), cmap='gray', vmin=0, vmax=1)
	plt.show()