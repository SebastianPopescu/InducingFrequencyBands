# Copyright (C) Secondmind Ltd 2021 - All Rights Reserved
# Unauthorised copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
"""
This module contains routines for finding a good initialisation point for kernel hyperparameters.
"""

from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from gpflow.models import GPModel
from pio_utilities import pio_logging

from pio_autogp.kernels.minecraft.transition import (
    SpectralNonstationaryCombination,
    SpectralTransitionWindow,
)

LOG = pio_logging.logger(__name__)


#TODO -- need to see what I can do with this


@tf.function
def try_random_hyperparameters(
    model: GPModel,
    nyquist_freqs: tf.Tensor,
    n_components: tf.Tensor,
    x_interval: tf.Tensor,
    y_variance: tf.Tensor,
) -> Tuple[tf.Tensor, Tuple]:
    """Evaluate the log likelihood for a random set of hyperparameters."""

    if isinstance(model.kernel, SpectralNonstationaryCombination):
        # Need to randomise window parameters in addition to the purely spectral ones
        # todo make robust to difference choice of x scaling
        random_position = tf.random.uniform([1], -2.0, 2.0, default_float())
        random_width = tf.random.uniform([1], 0.01, 2.0, default_float())
        model.kernel.transition_point.assign(random_position[0])
        model.kernel.transition_width.assign(random_width[0])

        if isinstance(model.kernel, SpectralTransitionWindow):
            random_window_width = tf.random.uniform([1], 0.01, 1.0, default_float())
            model.kernel.window_width.assign(random_window_width[0])

        random_correlation = tf.random.uniform([1], 0, 2, tf.dtypes.int32)
        random_correlation = tf.cast(random_correlation, default_float())

        model.kernel.correlation_coefficient.assign(random_correlation[0])

        params = randomise_initial_components(
            nyquist_freqs, n_components, x_interval, y_variance, do_secondary_variances=True
        )
        model.kernel.set_component_parameters(*params)

    else:
        params = randomise_initial_components(nyquist_freqs, n_components, x_interval, y_variance)
        model.kernel.set_component_parameters(*params)

    return model.maximum_log_likelihood_objective(), params


def randomise_initial_components(
    nyquist_freqs: np.ndarray,
    n_components: tf.Tensor,
    x_interval: tf.Tensor,
    y_variance: float,
    do_secondary_variances: bool = False,
) -> Tuple:
    """
    Initialise the spectral component parameters by making random draws from their prior.

    :param nyquist_freqs: The Nyquist frequency associated with each dimension:
       1/(2*min_x_separation) when min_x_separation != 0
       1. when min_x_separation == 0.
    :param n_components: The number of spectral components.
    :param x_interval: The maximum range spanned by the observations x.
    :param y_variance: The variance of the y data.
    :param do_secondary_variances: Whether to provide a second set of variances for Minecraft.
    :return: A tuple of means, bandwidths, and variances reflecting the proposed parameter values.
    """
    big_number = 1e10

    # Assumes ndims = 1
    n_bass_components = tf.math.floordiv(n_components, 2)
    n_treble_components = n_components - n_bass_components
    nyq_freq = nyquist_freqs[0]
    fundamental_freq = tf.math.reciprocal(x_interval)

    # fundamental_freq could be infinite when input data is degenerate.
    # In this case, we want to set reasonable defaults.
    default_min_treble = big_number * tf.ones_like(fundamental_freq)
    min_treble = tf.math.minimum(fundamental_freq, default_min_treble)
    max_treble = nyq_freq
    default_max_bass = tf.ones_like(fundamental_freq)
    max_bass = tf.math.minimum(fundamental_freq, default_max_bass)
    min_bass = max_bass / 1e8

    log_bass_freqs = tf.random.uniform(
        [n_bass_components], tf.math.log(min_bass), tf.math.log(max_bass), default_float()
    )
    bass_freqs = tf.exp(log_bass_freqs)
    treble_freqs = tf.random.uniform([n_treble_components], min_treble, max_treble, default_float())

    means = tf.concat((bass_freqs, treble_freqs), 0)
    means = tf.expand_dims(means, axis=0)

    b_mean = tf.math.log(x_interval) + tf.cast(0.3 * tf.random.normal(shape=[1]), default_float())
    lognormal = tfp.distributions.LogNormal(loc=0, scale=0.3)
    b_sigma = tf.cast(lognormal.sample([1]), default_float())
    b_lognormal = tfp.distributions.LogNormal(loc=b_mean, scale=b_sigma)
    inv_bandwidths = b_lognormal.sample(sample_shape=tf.shape(input=means))
    inv_bandwidths = tf.squeeze(inv_bandwidths, axis=2)  # weird behaviour of lognormal sampler

    # inv_bandwidths are infinite when the input data is degenerate (x_interval == 0)
    # In this case, we want to set reasonable defaults
    default_bandwidths = big_number * tf.ones_like(inv_bandwidths)
    bandwidths = tf.math.minimum(tf.abs(1 / inv_bandwidths), default_bandwidths)

    if do_secondary_variances:
        # todo generalise to n-variances
        variance_distribution = tfp.distributions.LogNormal(loc=1.0, scale=2.0)

        variances = tf.cast(
            variance_distribution.sample(sample_shape=tf.shape(tf.squeeze(means))), default_float(),
        )
        secondary_variances = tf.cast(
            variance_distribution.sample(sample_shape=tf.shape(tf.squeeze(means))), default_float(),
        )

        # normalise total
        variances = y_variance * variances / tf.reduce_sum(variances)
        secondary_variances = y_variance * secondary_variances / tf.reduce_sum(secondary_variances)

        return means, bandwidths, variances, secondary_variances

    else:
        variances = y_variance * tf.ones_like(tf.squeeze(means))
        variances /= tf.cast(n_components, default_float())

        random_params = (means, bandwidths, variances)

    return random_params
