# Copyright (C) Secondmind Ltd 2021 - All Rights Reserved
# Unauthorised copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
""" This module contains routines which assist in the development of spectral kernels. """
import math

import tensorflow as tf
from gpflow.config import default_float

TINY = 1e-20


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
