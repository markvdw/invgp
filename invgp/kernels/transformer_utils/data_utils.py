"""
MIT License

Copyright (c) 2017 Kevin Zakka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import time
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image


def write_hdf5(arr, outfile):
    """
    Write an numpy array to a file in HDF5 format.
    """
    with h5py.File(outfile, "w", libver='latest') as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


def load_hdf5(infile):
    """
    Load a numpy array stored in HDF5 format into a numpy array.
    """
    with h5py.File(infile, "r", libver='latest') as hf:
        return hf["image"][:]


def img_to_array(data_path, desired_size=None, view=False):
    """
    Util function for loading RGB image into 4D numpy array.

    Returns array of shape (1, H, W, C)

    References
    ----------
    - adapted from keras preprocessing/image.py
    """
    img = Image.open(data_path)
    img = img.convert('RGB')
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    if view:
        img.show()

    # preprocess    
    x = np.asarray(img, dtype='float32')
    x = np.expand_dims(x, axis=0)
    x /= 255.0

    return x


def array_to_img(x):
    """
    Util function for converting 4D numpy array to numpy array.

    Returns PIL RGB image.

    References
    ----------
    - adapted from keras preprocessing/image.py
    """
    x = np.asarray(x)
    x += max(-np.min(x), 0)
    x_max = np.max(x)
    if x_max != 0:
        x /= x_max
    x *= 255
    return Image.fromarray(x.astype('uint8'), 'RGB')


def run_op(x):
    """
    Utility function for debugging in tensorflow.

    Runs session to convert tensor to numpy array.
    """
    # intialize the variable
    init_op = tf.global_variables_initializer()

    # run the graph
    with tf.Session() as sess:
        sess.run(init_op)
        return sess.run(x)


def to_categorical(y, num_classes):
    """
    1- hot encodes a tensor.
    """
    return np.eye(num_classes, dtype='uint8')[y]
