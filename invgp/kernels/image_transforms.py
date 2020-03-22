import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from gpflow.config import default_float
from .transformer import spatial_transformer_network as stn


def rotate_img_angles(Ximgs, angles, interpolation_method):
    """
    :param Ximgs: Images to rotate.
    :param angles: Angles in degrees to rotate by.
    :param interpolation_method: Interpolation method.
    :return:
    """

    def rotate(angle):
        anglerad = tf.cast(angle / 180 * np.pi, tf.float32)
        return tf.reshape(
            tfa.image.rotate(Ximgs[:, :, :, None], anglerad, interpolation_method),
            (tf.shape(Ximgs)[0], tf.shape(Ximgs)[1] * tf.shape(Ximgs)[2])
        )

    return tf.transpose(tf.map_fn(rotate, angles, dtype=default_float()), (1, 0, 2))


def rotate_img_angles_stn(Ximgs, angles):
    """
    Uses spatial transformer networks to rotate a batch of images by different angles. The entire batch is rotated
    by all angles
    :param Ximgs: input images [None, H, W] or [None, H, W, 1]
    :param angles: angles in degrees to rotate by [P]
    :return: [None, P, H*W]
    """
    Ximgs = tf.cast(Ximgs, tf.float32)
    angles = tf.cast(angles, tf.float32)

    if len(Ximgs.get_shape()) == 3:
        Ximgs = tf.expand_dims(Ximgs, -1)  # [None, H, W, 1]

    def rotate(angle):
        # Prepare angle
        angle_rad = tf.cast(angle / 180 * np.pi, default_float())
        # Compute affine transformation (tile as per image)
        theta = tf.stack([tf.cos(angle_rad), -tf.sin(angle_rad), 0., tf.sin(angle_rad), tf.cos(angle_rad), 0.])
        theta = tf.reshape(theta, [1, -1])
        theta = tf.tile(theta, [tf.shape(Ximgs)[0], 1])

        return tf.reshape(
            tf.squeeze(stn(Ximgs, theta)), [tf.shape(Ximgs)[0], tf.shape(Ximgs)[1] * tf.shape(Ximgs)[2]]
        )  # [None, H*W]

    result = tf.transpose(tf.map_fn(rotate, angles, dtype=Ximgs.dtype), (1, 0, 2))
    return tf.cast(result, default_float())  # [None, P, H*W]


def _stn_theta_vec(angle_deg, sx, sy, tx, ty):
    """
    Compute 6-parameter theta vector from physical components
    :param angle_deg: rotation angle
    :param sx: scale in x direction
    :param sy: scale in y direction
    :param tx: shear in x direction
    :param ty: shear in y direction
    :return:
    """
    angle_rad = tf.cast(angle_deg / 180 * np.pi, default_float())
    s = tf.sin(angle_rad)
    c = tf.cos(angle_rad)

    return [sx * c - ty * sx * s, tx * sy * c - sy * s, 0., sx * s + ty * sx * c, tx * sy * s + sy * c, 0.]


def _apply_stn(Ximgs, theta):
    """
    Use spatial transformer networks to apply a general affine transformation (6 parameters). All images are transformed
    by the SAME theta
    :param Ximgs: [None, H, W, 1]
    :param theta: [6]
    :return: [None, H, W, 1]
    """
    theta = tf.reshape(theta, [1, -1])
    theta = tf.tile(theta, [tf.shape(Ximgs)[0], 1])

    return tf.reshape(tf.squeeze(stn(Ximgs, theta)), [tf.shape(Ximgs)[0], tf.shape(Ximgs)[1] * tf.shape(Ximgs)[2]])


def apply_stn_batch(Ximgs, thetas):
    """
    Use spatial transformer networks to apply a general affine transformation (6 parameters). Every image is transformed
    with every theta
    :param Ximgs: input images [None, H, W] or [None, H, W, 1]
    :param thetas: parameters of the affine transformation by [P, 6]
    :return: [None, P, H*W]
    """

    Ximgs = tf.cast(Ximgs, tf.float32)
    thetas = tf.cast(thetas, tf.float32)

    if len(Ximgs.get_shape()) == 3:
        Ximgs = tf.expand_dims(Ximgs, -1)  # [None, H, W, 1]

    result = tf.transpose(tf.map_fn(lambda x: _apply_stn(Ximgs, x), thetas, dtype=Ximgs.dtype), (1, 0, 2))
    return tf.cast(result, default_float())
