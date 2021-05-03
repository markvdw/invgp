from invgp.kernels import orbits
import matplotlib.pyplot as plt
import tensorflow as tf
from deepkernelinv_experiments.utils.load_datasets import load
import tensorflow_datasets as tfds
import numpy as np

#(X, Y), _ = load_mnist()

class args:
    dataset = "MNIST"
    image_shape = (32, 32, 3) if dataset == "CIFAR10" else (28, 28, 1)
    subset_size = None
    img_dim = 32*32*3 if dataset=='CIFAR10' else 28*28

train, test = load(args)
X, _ = list(zip(*[a for a in tfds.as_numpy(train.take(7).cache())]))
X = np.stack(X).astype('float64')
# plot 1 - X to be able to see the offset bug
X = 1 - X

orbit = orbits.InterpretableSpatialTransform(minibatch_size=2, theta_min=[0., 1., 1., 0., 0.], theta_max=[0., 1., 1., 0., 0.], input_dim=args.img_dim)
Xo = orbit(X[:2])

if args.dataset == "MNIST":
	print('plotting')
	fig, ax = plt.subplots(2, 2)
	# original images
	ax[0, 0].imshow(tf.reshape(X[0], (28, 28)), cmap='gray', vmin=0, vmax=1)
	ax[0, 1].imshow(tf.reshape(X[1], (28, 28)), cmap='gray', vmin=0, vmax=1)
	# transformed images
	ax[1, 0].imshow(tf.reshape(Xo[0, 0, :], (28, 28)), cmap='gray', vmin=0, vmax=1)
	ax[1, 1].imshow(tf.reshape(Xo[1, 0, :], (28, 28)), cmap='gray', vmin=0, vmax=1)
	plt.show()