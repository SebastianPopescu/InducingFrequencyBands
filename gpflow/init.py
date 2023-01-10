#import torch
#from . import gpr

import numpy as np
from scipy.special import gamma
from scipy.stats import multivariate_normal
from scipy import signal

#NOTE -- this is from pm_research
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

    powers = []

    for _ in range(n_components):

        mid_point_value = 0.5 * ( rbf_spectral_density(means[0,_] - 0.5 * bandwidths[0,_]) +
            rbf_spectral_density(means[0,_] + 0.5 * bandwidths[0,_]) )

        print(mid_point_value)
        powers.append(mid_point_value)
    print('-----------check this ----------------')
    print(powers)

    return means, bandwidths, powers


def get_lomb_scargle_value(X, Y, freq, maxfreq = None, transformed = False):

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

    powers = []

    for _ in range(n_components):

        mid_point_value = 0.5 * (  get_lomb_scargle_value(X = X, Y = Y, freq = np.array([means[0,_] - 0.5 * bandwidths[0,_]]) ) +
            get_lomb_scargle_value(X = X, Y = Y, freq = np.array([means[0,_] + 0.5 * bandwidths[0,_]]) ))

        print(mid_point_value)
        powers.append(mid_point_value)
    print('-----------check this ----------------')
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





#TODO -- eventually introduce this into the codebase
#def BNSE(x, y, y_err=None, max_freq=None, n=1000, iters=100):
"""
    Bayesian non-parametric spectral estimation [1] is a method for estimating the power spectral density of a signal that uses a Gaussian process with a spectral mixture kernel to learn the spectral representation of the signal. The resulting power spectral density is distributed as a generalized Chi-Squared distributions.

    Args:
        x (numpy.ndarray): Input data of shape (data_points,).
        y (numpy.ndarray): Output data of shape (data_points,).
        y_err (numpy.ndarray): Output std.dev. data of shape (data_points,).
        max_freq (float): Maximum frequency of the power spectral density. If not given the Nyquist frequency is estimated and used instead.
        n (int): Number of points in frequency space to sample the power spectral density.
        iters (int): Number of iterations used to train the Gaussian process.

    Returns:
        numpy.ndarray: Frequencies of shape (n,).
        numpy.ndarray: Power spectral density mean of shape (n,).
        numpy.ndarray: Power spectral density variance of shape (n,).

    [1] F. Tobar, Bayesian nonparametric spectral estimation, Advances in Neural Information Processing Systems, 2018
"""

"""
    x -= np.median(x)
    x_range = np.max(x)-np.min(x)
    x_dist = x_range/len(x)
    if max_freq is None:
        max_freq = 0.5/x_dist

    x = torch.tensor(x, device=gpr.config.device, dtype=gpr.config.dtype)
    if x.ndim == 0:
        x = x.reshape(1,1)
    elif x.ndim == 1:
        x = x.reshape(-1,1)
    y = torch.tensor(y, device=gpr.config.device, dtype=gpr.config.dtype).reshape(-1,1)

    kernel = gpr.SpectralKernel()
    model = gpr.Exact(kernel, x, y, data_variance=y_err**2 if y_err is not None else None)

    # initialize parameters
    magnitude = y.var()
    mean = 0.01
    variance = 0.25 / np.pi**2 / x_dist**2
    noise = y.std()/10.0
    model.kernel.magnitude.assign(magnitude)tfp.optimizer
    model.kernel.mean.assign(mean, upper=max_freq)
    model.kernel.variance.assign(variance)
    model.likelihood.scale.assign(noise)

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0)
    for i in range(iters):
        optimizer.step(model.loss)

    alpha = float(0.5/x_range**2)
    w = torch.linspace(0.0, max_freq, n, device=gpr.config.device, dtype=gpr.config.dtype).reshape(-1,1)

    def kernel_ff(f1, f2, magnitude, mean, variance, alpha):
        # f1,f2: MxD,  mean,variance: D
        mean = mean.reshape(1,1,-1)
        variance = variance.reshape(1,1,-1)
        gamma = 2.0*np.pi**2*variance
        const = 0.5 * np.pi * magnitude / torch.sqrt(alpha**2 + 2.0*alpha*gamma.prod())
        exp1 = -0.5 * np.pi**2 / alpha * gpr.Kernel.squared_distance(f1,f2)  # MxMxD
        exp2a = -2.0 * np.pi**2 / (alpha+2.0*gamma) * (gpr.Kernel.average(f1,f2)-mean)**2  # MxMxD
        exp2b = -2.0 * np.pi**2 / (alpha+2.0*gamma) * (gpr.Kernel.average(f1,f2)+mean)**2  # MxMxD
        return const * (torch.exp(exp1+exp2a) + torch.exp(exp1+exp2b)).sum(dim=2)

    def kernel_tf(t, f, magnitude, mean, variance, alpha):
        # t: NxD,  f: MxD,  mean,variance: D
        mean = mean.reshape(1,-1)
        variance = variance.reshape(1,-1)
        gamma = 2.0*np.pi**2*variance
        Lq_inv = np.pi**2 * (1.0/alpha + 1.0/gamma)  # 1xD
        Lq_inv = 1.0/Lq_inv  # this line must be kept, is this wrong in the paper?

        const = torch.sqrt(np.pi/(alpha+gamma.prod()))  # 1
        exp1 = -np.pi**2 * torch.tensordot(t**2, Lq_inv.T, dims=1)  # Nx1
        exp2a = -torch.tensordot(np.pi**2/(alpha+gamma), (f-mean).T**2, dims=1)  # 1xM
        exp2b = -torch.tensordot(np.pi**2/(alpha+gamma), (f+mean).T**2, dims=1)  # 1xM
        exp3a = -2.0*np.pi * torch.tensordot(t.mm(Lq_inv), np.pi**2 * (f/alpha + mean/gamma).T, dims=1)  # NxM
        exp3b = -2.0*np.pi * torch.tensordot(t.mm(Lq_inv), np.pi**2 * (f/alpha - mean/gamma).T, dims=1)  # NxM

        a = 0.5 * magnitude * const * torch.exp(exp1)
        real = torch.exp(exp2a)*torch.cos(exp3a) + torch.exp(exp2b)*torch.cos(exp3b)
        imag = torch.exp(exp2a)*torch.sin(exp3a) + torch.exp(exp2b)*torch.sin(exp3b)
        return a * real, a * imag

    with torch.no_grad():
        Ktt = kernel(x)
        Ktt += model.likelihood.scale().square() * torch.eye(x.shape[0], device=gpr.config.device, dtype=gpr.config.dtype)
        Ltt = model._cholesky(Ktt, add_jitter=True)

        Kff = kernel_ff(w, w, kernel.magnitude(), kernel.mean(), kernel.variance(), alpha)
        Pff = kernel_ff(w, -w, kernel.magnitude(), kernel.mean(), kernel.variance(), alpha)
        Kff_real = 0.5 * (Kff + Pff)
        Kff_imag = 0.5 * (Kff - Pff)

        Ktf_real, Ktf_imag = kernel_tf(x, w, kernel.magnitude(), kernel.mean(), kernel.variance(), alpha)

        a = torch.cholesky_solve(y,Ltt)
        b = torch.linalg.solve_triangular(Ltt,Ktf_real,upper=False)
        c = torch.linalg.solve_triangular(Ltt,Ktf_imag,upper=False)

        mu_real = Ktf_real.T.mm(a)
        mu_imag = Ktf_imag.T.mm(a)
        var_real = Kff_real - b.T.mm(b)
        var_imag = Kff_imag - c.T.mm(c)

        # The PSD equals N(mu_real,var_real)^2 + N(mu_imag,var_imag)^2, which is a generalized Chi-Squared distribution
        var_real = var_real.diagonal().reshape(-1,1)
        var_imag = var_imag.diagonal().reshape(-1,1)
        mu = mu_real**2 + mu_imag**2 + var_real + var_imag
        var = 2.0*var_real**2 + 2.0*var_imag**2 + 4.0*var_real*mu_real**2 + 4.0*var_imag*mu_imag**2

        w = w.cpu().numpy().reshape(-1)
        mu = mu.cpu().numpy().reshape(-1)
        var = var.cpu().numpy().reshape(-1)
    return w, mu, var
"""

