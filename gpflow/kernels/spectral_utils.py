# Copyright (C) Secondmind Ltd 2021 - All Rights Reserved
# Unauthorised copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
""" This module contains routines which assist in the development of spectral kernels. """
import math

import tensorflow as tf
from gpflow.config import default_float
from ..kernels import Matern12, Matern32, Matern52

TINY = 1e-20


def matern_spectral_density(freq, kernel):

    if isinstance(kernel, Matern12):

        lamb = tf.math.reciprocal(kernel.lengthscales)
        S = (2. * kernel.variance * lamb ) / (lamb**2 + freq**2)   
    
    elif isinstance(kernel, Matern32):
        pass
    elif isinstance(kernel, Matern52):
        pass

    return S 


def sinc(x, apply_normalisation=True):
    """
    Evaluate the sinc function, with special treatment required at very small values of x.
    The standard sinc is ``sin x / x``. Normalised sinc is ``sin (pi * x) / pi * x``.

    :param x: TODO missing param description
    :param apply_normalisation: TODO missing param description
    """

    tiny_x = tf.constant(TINY, dtype=default_float())
    y = tf.where(condition=tf.math.less(tf.abs(x), tiny_x), x=tiny_x, y=x)

    if apply_normalisation:
        y *= tf.constant(math.pi, dtype=x.dtype)

    return tf.sin(y) / y
