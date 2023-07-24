#import torch
#from . import gpr

import numpy as np
from scipy.special import gamma
from scipy.stats import multivariate_normal
from scipy import signal

#NOTE -- this is from pm_research -- Fergus
"""
def matern_spectral_density(freq, dim, nu, lengthscale, variance=1., freq_upper_bound=None):

    const_num = 2**dim * np.pi**(dim/2) * gamma(nu + dim/2) * (2 * nu)**nu
    const_den = gamma(nu) * lengthscale**(2 * nu)
    freq_term = 2 * nu / lengthscale**2 + 4 * np.pi**2 * np.sum((freq-1)**2, axis=1) #1.33
    freq_term_s = 2 * nu / lengthscale**2 + 4 * np.pi**2 * np.sum((freq+1)**2, axis=1)
    S = const_num / const_den * .5 * (freq_term**(-nu - dim/2) + freq_term_s**(-nu - dim/2))

    if freq_upper_bound is not None:
        abs_freq0 = np.abs(freq[:, 0])
        abs_freq1 = np.abs(freq[:, 1])

        beyond_max_freq = abs_freq0 > freq_upper_bound
        S[beyond_max_freq] = S[beyond_max_freq] * 0.

        # Repeat in other dimension
        beyond_max_freq = abs_freq1 > freq_upper_bound
        S[beyond_max_freq] = S[beyond_max_freq] * 0.

    return S * variance
"""

def riemann_approximate_rbf_initial_components(
    nyquist_freqs, n_components, x_interval):
    """
    Initialise the spectral component parameters so as to correspond to disjoint bandlimits
    which should ideally replicate the spectum of an RBF kernel given enough spectral blocks.

    :param nyquist_freqs: The Nyquist frequency associated with each dimension:
        1/(2*min_x_separation).
    :param n_components: The number of spectral components.
    :param x_interval: The maximum range spanned by the observations x.
    :return: A tuple of means, bandwidths, and variances reflecting the proposed parameter values.
    """
    #NOTE -- currently works just with 1D data??

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
        bandwidths = np.ones_like(means) * width
    means_list.append(means)
    bandwidths_list.append(bandwidths)

    means = np.stack(means_list) # expected shape (D, M)
    bandwidths = np.stack(bandwidths_list) # expected shape (D, M)    

    powers = []

    for _ in range(n_components):

        mid_point_value = 0.5 * ( rbf_spectral_density(means[0,_] - 0.5 * bandwidths[0,_]) +
            rbf_spectral_density(means[0,_] + 0.5 * bandwidths[0,_]) )
        powers.append(mid_point_value* (2. * bandwidths[0,_]))

    print('--- powers of Ind Freq Bands, should correspond to RBF spectral values ---')
    print(powers)

    return means, bandwidths, powers

def get_lomb_scargle_value(X, Y, freq, maxfreq = None, transformed = False):

    """
    TODO -- need to document this function, is it taken from Tobar's package?
    """

    n=10000
    X_scale = 1.
    if transformed:
        #Y = Y_transformer.forward(Y, X)
        pass

    idx = np.argsort(X[:,0])
    X = X[idx,0] * X_scale
    Y = Y[idx]

    nyquist = maxfreq
    if nyquist is None:
        dist = np.abs(X[1:]-X[:-1])
        nyquist = float(0.5 / np.average(dist))

    Y_freq_err = np.array([])

    X_freq = np.linspace(0.0, nyquist, n+1)[1:]
    #X_freq = np.linspace(0.0, nyquist, n+1)[1:]
    Y_freq = signal.lombscargle(X*2.0*np.pi, Y, X_freq)

    Y_freq_spec = signal.lombscargle(X*2.0*np.pi, Y, freq)

    #TODO -- I probably need to normalize this
    Y_freq_spec /= Y_freq.sum()*(X_freq[1]-X_freq[0]) # normalize

    return Y_freq_spec

def riemann_approximate_periodogram_initial_components(
    X, Y, nyquist_freqs, n_components, x_interval):
    """
    Initialise the spectral component parameters so as to correspond to disjoint bandlimits, 
    which should converge given enough spectral blocks to the periodogram of the data as obtained
    via the Lomb-Scargle method.

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
        bandwidths = np.ones_like(means) * width

    means_list.append(means)
    bandwidths_list.append(bandwidths)

    means = np.stack(means_list) # expected shape (D, M)
    bandwidths = np.stack(bandwidths_list) # expected shape (D, M)    

    powers = []

    for _ in range(n_components):

        mid_point_value = 0.5 * (get_lomb_scargle_value(X = X, Y = Y, 
                                                        freq = np.array([means[0,_] - 
                                                                         0.5 * bandwidths[0,_]]) ) +
            get_lomb_scargle_value(X = X, Y = Y, 
                                   freq = np.array([means[0,_] + 
                                                    0.5 * bandwidths[0,_]]) ))

        powers.append(mid_point_value * (2. * bandwidths[0,_]))
    print('--- powers of Ind Freq Bands, should correspond to Periodogram spectral values ---')
    print(powers)

    return means, bandwidths, powers

def neutral_initial_components(
    X, Y, nyquist_freqs, n_components, x_interval,  deltas):
    """
    Initialise the spectral component parameters so as to correspond to disjoint bandlimits,
    with a narrow pre-defined bandwidth.

    :param nyquist_freqs: The Nyquist frequency associated with each dimension:
        1/(2*min_x_separation). 
    :param n_components: The number of spectral components.
    :param x_interval: The maximum range spanned by the observations x.
    :param deltas: Dimension-wise bandwidth.
    :return: A tuple of means, bandwidths, and variances reflecting the proposed parameter values.
    """
    #NOTE -- currently works just with 1D data

    assert n_components > 0, "Require positive number of components"
    for _ in range(len(nyquist_freqs)):
        assert nyquist_freqs[_] > 0, "Nyquist frequencies should be positive for all dimensions"
    #for _ in range(len(deltas)):
    #    assert deltas[_] > 0, "General bandwidth should be positive for all dimensions"


    ndims = len(nyquist_freqs)
    variances_shape = n_components

    means_list = []
    bandwidths_list = []
    for i in range(ndims):

        nyq_freq = nyquist_freqs[i]
        fundamental_freq = 1 / x_interval[i]

        means, width = np.linspace(fundamental_freq, nyq_freq, n_components, retstep = True)
        #means = [ (m+1.) * deltas for m in range(n_components)]

        bandwidths = np.ones_like(means) * width

    means_list.append(means)
    bandwidths_list.append(bandwidths)

    means = np.stack(means_list) # expected shape (D, M)
    bandwidths = np.stack(bandwidths_list) # expected shape (D, M)    

    powers = []

    for _ in range(n_components):

        _value = 1.0 * (2. * bandwidths[0,_])
        powers.append(_value)
    
    print('--- powers of Ind Freq Bands, should be equally placed ---')
    print(powers)

    return means, bandwidths, powers


def rbf_spectral_density(freq, lengthscale = 0.301, variance=1.):

    constant_num = np.sqrt(np.sqrt(lengthscale)) / (2. * np.sqrt(np.pi))
    freq_term = np.exp(- np.sqrt(lengthscale) * freq**2 * 0.25)
    S =   constant_num * freq_term

    return S * variance

def matern_1_2_spectral_density(freq, lengthscale, variance=1.):

    freq_term = np.exp(- np.sqrt(lengthscale) * freq)
    S = freq_term

    return S * variance

def matern_3_2_spectral_density(freq, lengthscale, variance=1.):

    const_num = 1. + (np.sqrt(3.) * freq) / np.sqrt(lengthscale)
    freq_term = np.exp(- np.sqrt(3.) * np.sqrt(lengthscale) * freq)
    S = const_num * freq_term
    
    return S * variance

def matern_5_2_spectral_density(freq, lengthscale, variance=1.):

    const_num = 1. + (np.sqrt(5.) * freq) / np.sqrt(lengthscale) + (5. * freq**2)/(3. * lengthscale)
    freq_term = np.exp(- np.sqrt(5.) * np.sqrt(lengthscale) * freq)
    S = const_num * freq_term
    
    return S * variance

