# Copyright (C) Secondmind Ltd 2021 - All Rights Reserved
# Unauthorised copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
"""
This module contains NumPy routines for finding a good initialisation point for kernel
hyperparameters.
"""
from typing import List, Tuple

import numpy as np

def np_randomise_initial_components(
    nyquist_freqs: np.ndarray, n_components: int, x_interval: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialise the spectral component parameters by making random draws from their prior.

    :param nyquist_freqs: The Nyquist frequency associated with each dimension:
        1/(2*min_x_separation).
    :param n_components: The number of spectral components.
    :param x_interval: The maximum range spanned by the observations x.
    :return: A tuple of means, bandwidths, and variances reflecting the proposed parameter values.
    """

    assert n_components > 0, "Require positive number of components"
    assert np.all(nyquist_freqs > 0), "Nyquist frequencies should be positive"

    ndims = np.size(nyquist_freqs)
    variances_shape = n_components

    means_list = []

    for i in range(ndims):
        n_bass_components = n_components // 2 
        n_treble_components = n_components - n_bass_components
        nyq_freq = nyquist_freqs[i]
        fundamental_freq = 1 / x_interval

        min_treble = fundamental_freq
        max_treble = nyq_freq
        max_bass = fundamental_freq
        min_bass = max_bass / 1e8

        bass_freqs = np.exp(
            np.random.uniform(np.log(min_bass), np.log(max_bass), size=n_bass_components)
        )
        treble_freqs = np.random.uniform(min_treble, max_treble, size=n_treble_components)

        means = np.concatenate((bass_freqs, treble_freqs))

        means_list.append(means)

    means = np.stack(means_list)
    variances = np.ones(shape=variances_shape) / n_components
    bandwidths = 2.0 * means

    return means, bandwidths, variances



def np_disjoint_initial_components(
    nyquist_freqs: List[float], n_components: int, x_interval: List[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialise the spectral component parameters so as to correspond to disjoint bandlimits.

    :param nyquist_freqs: The Nyquist frequency associated with each dimension:
        1/(2*min_x_separation).
    :param n_components: The number of spectral components.
    :param x_interval: The maximum range spanned by the observations x.
    :return: A tuple of means, bandwidths, and variances reflecting the proposed parameter values.
    """
    #NOTE -- currently works just with 1D data


    assert n_components > 0, "Require positive number of components"
    for _ in range(len(nyquist_freqs)):
        assert nyquist_freqs[_] > 0, "Nyquist frequencies should be positive for all dimensions"

    ndims = len(nyquist_freqs)
    variances_shape = n_components

    means_list = []
    bandwidths_list = []
    for i in range(ndims):

        nyq_freq = nyquist_freqs[i]
        fundamental_freq = 1 / x_interval[i]

        means, width = np.linspace(fundamental_freq, nyq_freq, n_components, retstep = True)
        bandwidths = np.ones_like(means) * (width / 2.)

    means_list.append(means)
    bandwidths_list.append(bandwidths)

    means = np.stack(means_list) # expected shape (D, M)
    bandwidths = np.stack(bandwidths_list) # expected shape (D, M)    

    powers = [1.0 for _ in range(means.shape[1])]

    return means, bandwidths, powers
