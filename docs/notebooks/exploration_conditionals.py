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
from gpflow.inducing_variables import SpectralInducingPoints
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

N = 100  # Number of training observations

X_cond = np.linspace(0, 1.0, N)  # X values
Y_cond = func(X_cond) + 0.2 * rng.randn(N, )  # Noisy Y values

# %% [markdown]
# # Can choose between 'GPR' and 'SGPR'
MODEL = 'SGPR'
# %% [markdown]
# # Can choose between 'multisinc' (i.e., corresponds to symmetrical rectangular blocks for both
# # Inducing Frequency Bands and GP-MultiSinc), 'cosine' (i.e., dirac delta functions in the kernel spectrum)
## and Matern12 (i.e., corresponds to L2 features VFF)
KERNEL = 'Matern12' #'multisinc' #'cosine'
MAXFREQ = 10.
if KERNEL =='cosine':
    N_COMPONENTS = 1
else:
    N_COMPONENTS = 500
MAXITER = 1000

# %% [markdown]
# # Can choose between 'rbf', 'Periodogram' or 'Neutral', 'Kernel'
INIT_METHOD = 'Kernel'
#INIT_METHOD = 'rbf'
#INIT_METHOD = 'Periodogram' 
#INIT_METHOD ='Neutral'
DELTAS = 1e-1
#NOTE -- alpha needs to be set to a very low value, i.e., close to 0.
ALPHA = 1e-24
# %% [markdown]
# # Can choose between 'RKHS', 'L2', ''. Only valid for Matern12 kernels within the VFF framework.
FEATURES = 'L2' 
#FEATURES = 'RKHS' 
#FEATURES = ''

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

elif KERNEL == 'Matern12':
    #NOTE -- these will just be used as dummy variables since they are not needed
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

powers_np = [np.float64(np_float) for np_float in powers_np]

print('powers_np')
print(powers_np)


if KERNEL=='cosine':
    kern = gpflow.kernels.SpectralDiracDeltaBlock(means= means_np, 
    bandwidths= bandwidths_np, powers=powers_np, alpha = 0.)
elif KERNEL=='multisinc':
    kern = gpflow.kernels.MultipleSpectralBlock(n_components=N_COMPONENTS, means= means_np, 
        bandwidths= bandwidths_np, powers=powers_np, alpha=ALPHA)
elif KERNEL=='Matern12':

    a = -0.5
    b = 1.5
    #N_COMPONENTS - Number of spectral inducing points
    # creating the harmonics
    ms = np.arange(N_COMPONENTS) # this should be the usual ms for general experiments
    omegas = (2.0 * np.pi * ms) / (b - a)

    if FEATURES=='RKHS':
        kern = gpflow.kernels.RKHS_Matern12(variance = 1.0, lengthscales= (b - a)/10.)
    elif FEATURES=='L2':
        kern = gpflow.kernels.L2_Matern12(variance = 1.0, lengthscales= (b - a)/10.)

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
    ax: Axes, kernel: gpflow.kernels.Kernel, *, optimise: bool = True, parametric: bool = True,
) -> None:
    #X = np.array([[-0.5], [0.0], [0.4], [0.5]])
    #Y = np.array([[1.0], [0.0], [0.6], [0.4]])
    

    if MODEL=='GPR':
        model = gpflow.models.GPR(
            (X_cond, Y_cond), kernel=deepcopy(kernel), noise_variance=1e-3
        )
    elif MODEL=='SGPR':
        
        if KERNEL=='Matern12':
            ind_pts = SpectralInducingPoints(a = a, b = b, omegas = omegas)
            model = gpflow.models.SGPR(
                data = (X_cond, Y_cond), kernel=deepcopy(kernel), inducing_variable = ind_pts, noise_variance=1e-3
            )

        else:
            ind_var = gpflow.inducing_variables.RectangularSpectralInducingPoints(kern = kernel)
            model = gpflow.models.SGPR(
                data = (X_cond, Y_cond), kernel=deepcopy(kernel), inducing_variable = ind_var, noise_variance=1e-3
            )

    if optimise:
        gpflow.set_trainable(model.likelihood, False)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)

    Xplot = np.linspace(a, b, 100)[:, None]

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

    MAXFREQ=15.
    data_object.plot_spectrum(ax = ax, maxfreq=MAXFREQ)

    for _ in range(N_COMPONENTS):
        #spectral_block_1a = make_component_spectrum(np.linspace(0, MAXFREQ, 1000), 
        #                                            means_np[:,_], bandwidths_np[:,_], 
        #                                            powers_np[_], use_blocks = True)
        #ax.plot(np.linspace(0, MAXFREQ, 1000), spectral_block_1a.ravel(), 
        #        label='$S_{aa}(\\nu)$', linewidth=.8)

        spectral_block_1a = make_component_spectrum(np.linspace(0, MAXFREQ, 1000), 
            tf.convert_to_tensor(kern.means[:,_]).numpy(), 
            tf.convert_to_tensor(kern.bandwidths[:,_]).numpy(), 
            tf.convert_to_tensor(kern.powers[_]).numpy(), 
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
        if KERNEL =='Matern12':
            pass
        else:
            plot_spectrum_blocks(spectrum_ax, data_object)
    elif MODEL=='SGPR':
        if KERNEL =='Matern12':
            _, (samples_ax, prediction_parametric_ax, prediction_non_parametric_ax, 
                spectrum_ax) = plt.subplots(nrows=4, ncols=1, figsize=(15, 10 * 4))
        else:
            _, (samples_ax, prediction_parametric_ax, prediction_non_parametric_ax, 
                spectrum_ax) = plt.subplots(nrows=3, ncols=1, figsize=(15, 10 * 3))
        plot_kernel_samples(samples_ax, kernel)
        plot_kernel_prediction(prediction_parametric_ax, kernel, optimise=optimise)
        plot_kernel_prediction(prediction_non_parametric_ax, kernel, optimise=optimise, 
                               parametric=False)
        if KERNEL =='Matern12':
            pass
        else:
            plot_spectrum_blocks(spectrum_ax, data_object)

    plt.tight_layout()
    plt.savefig(f'./figures/{MODEL}_{KERNEL}_{FEATURES}_{INIT_METHOD}_exploration.png')
    plt.close()

#TODO -- optimise = True seems to encounter numerical issues for L2 features VFF.
plot_kernel(kern, optimise = False)