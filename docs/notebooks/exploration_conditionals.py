# ---
# jupyter:
#   jupytext:
#     formats: ipynb,.pct.py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # This notebook explores function samples from certain spectral kernels 
# # (symmetrical rectangular blocks that are either initialized to be similar
# # to an RBF kernel, periodogram of data or just constant energy for each bandwidth).
# # We plot the conditional function (either based on some data for GPR model, or based
# # on inducing points for SGPR). 

# %%
# %matplotlib inline
import itertools
import time

import numpy.random as rnd

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tikzplotlib
from scipy.stats import norm

import matplotlib.pyplot as plt
plt.ion()
plt.close('all')
plt.style.use("ggplot")

import gpflow
from gpflow.ci_utils import reduce_in_tests
from gpflow.init import riemann_approximate_periodogram_initial_components, rbf_spectral_density, neutral_initial_components, riemann_approximate_rbf_initial_components
from gpflow.kernels.initialisation_spectral_np import np_disjoint_initial_components

# for reproducibility of this notebook:
rng = np.random.RandomState(123)
tf.random.set_seed(42)

from copy import deepcopy
from matplotlib.axes import Axes
from matplotlib.cm import coolwarm

def func(x):
    return (
        np.sin(x * 3 * 3.14)
        + 0.3 * np.cos(x * 9 * 3.14)
        + 0.5 * np.sin(x * 7 * 3.14)
    )

N = 120  # Number of training observations

X_cond = np.linspace(0, 1.0, N)  # X values
Y_cond = func(X_cond) + 0.2 * rng.randn(N, )  # Noisy Y values

# %% [markdown]
# # Can choose between 'GPR' and 'SGPR'
MODEL = 'SGPR'
# %% [markdown]
# # Can choose between 'multisinc' (i.e., corresponds to symmetrical rectangular blocks for both
# # Inducing Frequency Bands and GP-MultiSinc), 'cosine' (i.e., dirac delta functions in the kernel spectrum)
# # ,'decomposed_multisinc' (i.e., corresponds to asymmetrical rectangular blocks for
# # Inducing Frequency Bands, which essentially translates to cosine and sine transforms as the features) and
# # 'decomposed_dirac_multisinc' (i.e, corresponds to asymmetrical rectangular blocks which happen to be
# # Dirac Delta) that correspond to multi cosine kernels.
KERNEL = 'decomposed_dirac_multisinc'
#KERNEL = 'decomposed_multisinc'
#KERNEL = 'multisinc' 
#KERNEL = 'cosine'

MAXFREQ = 10.
if KERNEL =='cosine':
    N_COMPONENTS = 1
else:
    N_COMPONENTS = 50
MAXITER = 1000

# %% [markdown]
# # Can choose between 'rbf', 'Periodogram' or 'Neutral'
INIT_METHOD = 'rbf'
#INIT_METHOD = 'Periodogram' 
#INIT_METHOD ='Neutral'
DELTAS = 1e-1
#NOTE -- alpha needs to be set to a very low value, i.e., close to 0.
ALPHA = 1e-1

if KERNEL == 'cosine':
    #NOTE -- these will just be used as dummy variables in this initaliation
    means_np, bandwidths_np, powers_np = neutral_initial_components(
        X_cond.reshape((-1,1)), 
        Y_cond.ravel(), 
        [MAXFREQ], 
        n_components=N_COMPONENTS,
        x_interval = [X_cond.max() -  X_cond.min()],
        deltas = DELTAS,                                                                   
        )
    #NOTE -- let's take a lower frequency
    means_np = np.ones_like(means_np) * 5.0
    bandwidths_np = np.ones_like(bandwidths_np) * 1e-2
    powers_np = np.ones_like(bandwidths_np) * 1.0

elif INIT_METHOD == 'Periodogram':

    means_np, bandwidths_np, powers_np = riemann_approximate_periodogram_initial_components(
        X_cond.reshape((-1,1)), 
        Y_cond.ravel(), 
        [MAXFREQ], 
        n_components=N_COMPONENTS, 
        x_interval = [X_cond.max() -  X_cond.min()]
        )
elif INIT_METHOD == 'Neutral':

    means_np, bandwidths_np, powers_np = neutral_initial_components(
        X_cond.reshape((-1,1)), 
        Y_cond.ravel(), 
        [MAXFREQ], 
        n_components=N_COMPONENTS,
        x_interval = [X_cond.max() -  X_cond.min()],
        deltas = DELTAS,                                                                   
        )
elif INIT_METHOD == 'rbf':
    
    means_np, bandwidths_np, powers_np = riemann_approximate_rbf_initial_components(
        [MAXFREQ], 
        n_components=N_COMPONENTS, 
        x_interval = [X_cond.max() -  X_cond.min()]
        )

X_cond = X_cond.reshape((-1, 1))
Y_cond = Y_cond.reshape((-1, 1))

data_object = gpflow.data.Data(X = X_cond, Y = Y_cond)

means_np = means_np.astype(np.float)
bandwidths_np = bandwidths_np.astype(np.float64)

print('means_np')
print(means_np)
print(means_np.shape)

print('bandwidths_np')
print(bandwidths_np)
print(bandwidths_np.shape)

print('lower limits')
print(means_np - bandwidths_np * 0.5)

print('upper limits')
print(means_np + bandwidths_np * 0.5)


if KERNEL=='decomposed_multisinc' or KERNEL=='decomposed_dirac_multisinc':
    #NOTE -- we are just splitting in half the powers of each block
    # and assign them to the co/sine rectangular blocks.
    powers_np = [np.float64(np_float/2.) for np_float in powers_np]
else:
    powers_np = [np.float64(np_float) for np_float in powers_np]

print('powers_np')
print(powers_np)

if KERNEL=='cosine':
    kern = gpflow.kernels.SpectralDiracDeltaBlock(means= means_np, 
                                                  bandwidths= bandwidths_np, 
                                                  powers=powers_np, 
                                                  alpha = 0.)
elif KERNEL=='multisinc':
    kern = gpflow.kernels.MultipleSpectralBlock(n_components=N_COMPONENTS, 
                                                means= means_np, 
                                                bandwidths= bandwidths_np, 
                                                powers=powers_np, 
                                                alpha=ALPHA)
elif KERNEL=='decomposed_multisinc':
    kern = gpflow.kernels.DecomposedMultipleSpectralBlock(n_components=N_COMPONENTS, 
                                                          means= means_np, 
                                                          bandwidths= bandwidths_np, 
                                                          real_powers= powers_np, 
                                                          img_powers = powers_np,
                                                          alpha=ALPHA)
elif KERNEL=='decomposed_dirac_multisinc':
    kern = gpflow.kernels.DecomposedMultipleDiracSpectralBlock(n_components=N_COMPONENTS, 
                                                               means = means_np, 
                                                               bandwidths = bandwidths_np, 
                                                               real_powers= powers_np, 
                                                               img_powers = powers_np,
                                                               alpha=ALPHA)

def plot_kernel_samples(ax: Axes, kernel: gpflow.kernels.SpectralKernel) -> None:
    X = np.zeros((0, 1))
    Y = np.zeros((0, 1))
    model = gpflow.models.GPR((X, Y), kernel=deepcopy(kernel))
    Xplot = np.linspace(-1.0, 1.0, 100)[:, None]
    tf.random.set_seed(42)
    n_samples = 3
    # predict_f_samples draws n_samples examples of the function f, and returns their values at Xplot.
    #NOTE -- do I need to draw samples from the Nystrom approximation when MODEL=='SGPR'?
    fs = model.predict_f_samples(Xplot, n_samples)
    ax.plot(Xplot, fs[:, :, 0].numpy().T, label=kernel.__class__.__name__)
    ax.set_ylim(bottom=-5.0, top=5.0)
    ax.set_title("Example $f$s")


def plot_kernel_prediction(
    ax: Axes, kernel: gpflow.kernels.Kernel, *, optimise: bool = False, parametric: bool = True,
) -> None:
    #X = np.array([[-0.5], [0.0], [0.4], [0.5]])
    #Y = np.array([[1.0], [0.0], [0.6], [0.4]])
    
    if MODEL=='GPR':
        model = gpflow.models.GPR(
            (X_cond, Y_cond), kernel=deepcopy(kernel), noise_variance=1e-3
        )
    elif MODEL=='SGPR':
        if KERNEL=='multsinc':
            ind_var = gpflow.inducing_variables.SymRectangularSpectralInducingPoints(kern = kernel)
        elif KERNEL=='decomposed_multisinc':
            ind_var = gpflow.inducing_variables.AsymRectangularSpectralInducingPoints(kern = kernel)
        elif KERNEL=='decomposed_dirac_multisinc':
            ind_var = gpflow.inducing_variables.AsymDiracSpectralInducingPoints(kern = kernel)
        model = gpflow.models.SGPR(
            data = (X_cond, Y_cond), kernel=deepcopy(kernel), inducing_variable = ind_var, noise_variance=1e-3
        )

    if optimise:
        gpflow.set_trainable(model.likelihood, False)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)

    Xplot = np.linspace(-5.5, 5.5, 100)[:, None]

    if MODEL=='GPR':
        f_mean, f_var = model.predict_f(Xplot, full_cov=False)
        f_lower = f_mean - 1.96 * np.sqrt(f_var)
        f_upper = f_mean + 1.96 * np.sqrt(f_var)
    elif MODEL=='SGPR':
        #NOTE -- just the non-parametric part for the moment
        #f_mean, f_var = model.predict_f_non_parametric(Xplot, full_cov=False)
        if parametric:
            f_mean, f_var = model.predict_f_parametric(Xplot, full_cov=False)
        else:
            f_mean, f_var = model.predict_f_non_parametric(Xplot, full_cov=False)
        f_lower = f_mean - 1.96 * np.sqrt(f_var)
        f_upper = f_mean + 1.96 * np.sqrt(f_var)
    else:
        pass

    ax.scatter(X_cond, Y_cond, color="black")
    (mean_line,) = ax.plot(Xplot, f_mean, "-", label=kernel.__class__.__name__)
    color = mean_line.get_color()
    ax.plot(Xplot, f_lower, lw=0.1, color=color)
    ax.plot(Xplot, f_upper, lw=0.1, color=color)
    ax.fill_between(
        Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color=color, alpha=0.1
    )
    ax.set_ylim(bottom=-2.5, top=2.5)
    if MODEL=='SGPR' and parametric:
        ax.set_title("Example parametric data fit")
    elif MODEL=='SGPR' and not parametric:
        ax.set_title("Example non-parametric data fit")
    elif MODEL=='GPR':
        ax.set_title("Example data fit")

def spectral_basis(x, mean, bandwidth, variance, use_blocks=True):
    if use_blocks:
        spectrum = np.where(abs(x - mean) <= 0.5 * bandwidth, 1./ (2. * bandwidth), 0.)
    else:
        spectrum = norm.pdf(x, mean, bandwidth)
    return variance * spectrum

def make_component_spectrum(x, mean, bandwidth, variance, use_blocks=True):
    
    spectrum = 1. * (spectral_basis(x, mean, bandwidth, variance, use_blocks) + 
                     spectral_basis(x, -mean, bandwidth, variance, use_blocks))
    
    return spectrum

def plot_spectrum_blocks(
    ax: Axes, data_object,
) -> None:
    # Periodogram and ''optimized'' symmetrical rectangles

    data_object.plot_spectrum(ax = ax, maxfreq=MAXFREQ)

    for _ in range(N_COMPONENTS):
        #spectral_block_1a = make_component_spectrum(np.linspace(0, MAXFREQ, 1000), 
        #                                            means_np[:,_], bandwidths_np[:,_], 
        #                                            powers_np[_], use_blocks = True)
        #ax.plot(np.linspace(0, MAXFREQ, 1000), spectral_block_1a.ravel(), 
        #        label='$S_{aa}(\\nu)$', linewidth=.8)

        if KERNEL=='decomposed_multisinc' or KERNEL=='decomposed_dirac_multisinc':
            _kernel_powers = kern.real_powers[_] + kern.img_powers[_]
        else:
            _kernel_powers = kern.powers[_]
        spectral_block_1a = make_component_spectrum(np.linspace(0, MAXFREQ, 1000), 
            tf.convert_to_tensor(kern.means[:,_]).numpy(), 
            tf.convert_to_tensor(kern.bandwidths[:,_]).numpy(), 
            tf.convert_to_tensor(_kernel_powers).numpy(), 
            use_blocks = True)

        ax.plot(np.linspace(0, MAXFREQ, 1000), spectral_block_1a, label='SymBand_'+str(_), 
                linewidth=.8)


def plot_kernel(
    kernel: gpflow.kernels.Kernel, *, optimise: bool = False
) -> None:
    if MODEL=='GPR':
        _, (samples_ax, prediction_ax, spectrum_ax) = plt.subplots(nrows=3, ncols=1, figsize=(15, 10 * 3))
        plot_kernel_samples(samples_ax, kernel)
        plot_kernel_prediction(prediction_ax, kernel, optimise=optimise)
        plot_spectrum_blocks(spectrum_ax, data_object)
    elif MODEL=='SGPR':
        _, (samples_ax, prediction_parametric_ax, prediction_non_parametric_ax, 
            spectrum_ax) = plt.subplots(nrows=4, ncols=1, figsize=(15, 10 * 4))
        plot_kernel_samples(samples_ax, kernel)
        plot_kernel_prediction(prediction_parametric_ax, kernel, optimise=optimise)
        plot_kernel_prediction(prediction_non_parametric_ax, kernel, optimise=optimise, parametric=False)
        plot_spectrum_blocks(spectrum_ax, data_object)

    plt.tight_layout()
    plt.savefig(f'./figures/{MODEL}_{KERNEL}_{INIT_METHOD}_exploration.png')
    plt.close()

plot_kernel(kern)


def plot_covariance(kernel):
    """
    #TODO -- write documentation
    """
    if MODEL=='GPR':
        model = gpflow.models.GPR(
            (X_cond, Y_cond), kernel=deepcopy(kernel), noise_variance=1e-3
        )
    elif MODEL=='SGPR':
        if KERNEL=='multsinc':
            ind_var = gpflow.inducing_variables.SymRectangularSpectralInducingPoints(kern = kernel)
        elif KERNEL=='decomposed_multisinc':
            ind_var = gpflow.inducing_variables.AsymRectangularSpectralInducingPoints(kern = kernel)
        elif KERNEL=='decomposed_dirac_multisinc':
            ind_var = gpflow.inducing_variables.AsymDiracSpectralInducingPoints(kern = kernel)
        model = gpflow.models.SGPR(
            data = (X_cond, Y_cond), kernel=deepcopy(kernel), inducing_variable = ind_var, noise_variance=1e-3
        )

    _kuf, _kuu = model.get_covariances()

    print(_kuf)
    print(_kuu)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 10))

    # Kuf
    ax1.matshow(_kuf.numpy(), aspect="auto")
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.set_title("Kuf")

    # Kuu
    ax2.matshow(_kuu.numpy(), aspect="auto")
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.set_title("Kuu")

    plt.savefig(f'./figures/{MODEL}_{KERNEL}_{INIT_METHOD}_covariances.png')
    plt.close()

plot_covariance(kern)