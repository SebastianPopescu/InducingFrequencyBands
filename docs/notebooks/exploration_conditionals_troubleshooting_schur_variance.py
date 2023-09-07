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
from gpflow.init import (
    riemann_approximate_periodogram_initial_components, 
    rbf_spectral_density, 
    neutral_initial_components, 
    riemann_approximate_rbf_initial_components
)
from gpflow.kernels.initialisation_spectral_np import np_disjoint_initial_components
import statsmodels.api as sm
# for reproducibility of this notebook:
rng = np.random.RandomState(123)
tf.random.set_seed(42)

from copy import deepcopy
from matplotlib.axes import Axes
from matplotlib.cm import coolwarm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_inducing", type = int, help="spectral components")
args = parser.parse_args()

command = f'mkdir -p ./figures_troubleshooting/num_ind_{args.num_inducing}'
os.system(command)


def func(x):
    return (
        np.sin(x * 3 * 3.14)
        + 0.3 * np.cos(x * 9 * 3.14)
        + 0.5 * np.sin(x * 7 * 3.14)
    )

N = 500  # Number of training observations

EXPERIMENT = 'toy_sine'
#EXPERIMENT = 'hr1'
#EXPERIMENT = 'hr2'
#EXPERIMENT = 'sunspots'

if EXPERIMENT=='toy_sine':
    #X_cond = np.linspace(0, 1.0, N)  # X values
    X_cond = rng.rand(N, 1) * 2 - 1  # X values
    Y_cond = func(X_cond) + 0.2 * rng.randn(N, 1)  # Noisy Y values

elif EXPERIMENT=='hr1':
    Y_cond = np.loadtxt('./docs/notebooks/data/hr2.txt') 
    X_cond = (np.linspace(0, 1800,1800))
    time_label = 'time'
    signal_label = 'heart-rate signal'

elif EXPERIMENT == 'hr2':
    Y_cond = np.loadtxt('./docs/notebooks/data/hr1.txt') 
    X_cond = (np.linspace(0, 1800,1800))
    time_label = 'time'
    signal_label = 'heart-rate signal'

elif EXPERIMENT == 'sunspots':
    dta = sm.datasets.sunspots.load_pandas().data
    Y_cond = np.array(dta.SUNACTIVITY)
    X_cond = np.array(dta.YEAR)
    time_label = 'time'
    signal_label = 'sunspot data'
        
X_offset = np.median(X_cond)
X_cond = X_cond - X_offset
Y_offset = np.mean(Y_cond)
Y_cond = Y_cond - Y_offset

MODEL = 'SGPR'
# %% [markdown]
# # 'SqExp' (i.e., corresponds to  
# # Dirac Delta rectangular blocks for
# # Integrated Fourier Features based on the underlying
# # squared exponential kernel) 

KERNEL = 'SqExp'
EPSILON = 1e-1

if EXPERIMENT=='toy_sine':
    MAXFREQ = 10.
    X_plotting_offset = 2.5

elif EXPERIMENT=='hr1':
    MAXFREQ = 0.03
    X_plotting_offset = 500

elif EXPERIMENT == 'hr2':
    MAXFREQ = 0.03
    X_plotting_offset = 500

elif EXPERIMENT == 'sunspots':
    MAXFREQ = 0.2
    X_plotting_offset = 500

N_COMPONENTS = args.num_inducing
MAXITER = 100

X_cond = X_cond.reshape((-1, 1))
Y_cond = Y_cond.reshape((-1, 1))

data_object = gpflow.data.Data(X = X_cond, Y = Y_cond)


kern = gpflow.kernels.SquaredExponential(variance=1.0, lengthscales=0.301)

ind_var = gpflow.inducing_variables.IFFRectangularSpectralInducingPoints(M = N_COMPONENTS, eps = EPSILON)

model = gpflow.models.SGPR(data = (X_cond, Y_cond), 
                           kernel=kern, 
                           inducing_variable = ind_var, 
                           noise_variance=1e-3)

#optimise = True
optimise = False

if optimise:
    #NOTE -- be careful with these deactivations
    #gpflow.set_trainable(model.likelihood, False)
    gpflow.set_trainable(model.inducing_variable.Z, False)

    opt_logs = gpflow.optimizers.Scipy().minimize(
        model.training_loss,
        variables=model.trainable_variables,
        method="l-bfgs-b",
        options={"disp": True, "maxiter": MAXITER})

def plot_kernel_samples(ax: Axes, kernel: gpflow.kernels.SpectralKernel) -> None:
    X = np.zeros((0, 1))
    Y = np.zeros((0, 1))
    model = gpflow.models.GPR((X, Y), kernel=kernel)
    Xplot = np.linspace(-X_plotting_offset, X_plotting_offset, 100)[:, None]
    tf.random.set_seed(42)
    n_samples = 3
    # predict_f_samples draws n_samples examples of the function f, and returns their values at Xplot.
    #NOTE -- do I need to draw samples from the Nystrom approximation when MODEL=='SGPR'?
    fs = model.predict_f_samples(Xplot, n_samples)
    ax.plot(Xplot, fs[:, :, 0].numpy().T, label=kernel.__class__.__name__)
    ax.set_ylim(bottom=-5.0, top=5.0)
    ax.set_title("Example $f$s")

def plot_kernel_prediction(
    ax: Axes, model, *, optimise: bool = True, 
    parametric: bool = True,
    Qff: bool = False
) -> None:
    #X = np.array([[-0.5], [0.0], [0.4], [0.5]])
    #Y = np.array([[1.0], [0.0], [0.6], [0.4]])

    Xplot = np.linspace(np.min(X_cond) - X_plotting_offset, np.max(X_cond) + X_plotting_offset, 100)[:, None]

    if MODEL=='GPR':
        f_mean, f_var = model.predict_y(Xplot, full_cov=False)
        f_lower = f_mean - 1.96 * np.sqrt(f_var)
        f_upper = f_mean + 1.96 * np.sqrt(f_var)
    elif MODEL=='SGPR':
        #NOTE -- just the non-parametric part for the moment
        #f_mean, f_var = model.predict_f_non_parametric(Xplot, full_cov=False)
        if parametric:
            f_mean, f_var = model.predict_f_parametric(Xplot, full_cov=False)
        elif not parametric and Qff:
            f_mean, f_var = model.predict_f_non_parametric_Qff(Xplot, full_cov=False)
        elif not parametric and not Qff:
            f_mean, f_var = model.predict_f_non_parametric_Kff(Xplot, full_cov=False)
        f_lower = f_mean - 1.96 * np.sqrt(f_var)
        f_upper = f_mean + 1.96 * np.sqrt(f_var)
    else:
        pass

    ax.plot(X_cond, Y_cond,'.r', markersize=10, label='observations')
    #ax.scatter(X_cond, Y_cond, color="black")
    ax.plot(Xplot, f_mean, "-", color='blue', label=model.kernel.__class__.__name__)
    ax.plot(Xplot, f_lower, lw=0.1, color='blue')
    ax.plot(Xplot, f_upper, lw=0.1, color='blue')
    ax.fill_between(
        Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color='blue', alpha=0.1
    )
    
    #ax.set_ylim(bottom=-2.5, top=2.5)
    if MODEL=='SGPR' and parametric:
        ax.set_title("Example parametric data fit")
    elif MODEL=='SGPR' and not parametric and Qff:
        ax.set_title("Example non-parametric data fit -- Qff")
    elif MODEL=='SGPR' and not parametric and not Qff:
        ax.set_title("Example non-parametric data fit -- Kff")
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
    ax: Axes, model, data_object,
) -> None:
    # Periodogram and ''optimized'' symmetrical rectangles

    data_object.plot_spectrum(ax = ax, maxfreq=MAXFREQ)


    #TODO -- I can reintroduce this a bit later.
    """
    for _ in range(N_COMPONENTS):
        
        _kernel_powers = model.kernel.powers[_]

        spectral_block_1a = make_component_spectrum(np.linspace(0, MAXFREQ, 1000), 
            tf.convert_to_tensor(model.inducing_variable.Z[:,_]).numpy(), 
            tf.convert_to_tensor(model.inducing_variable[:,_]).numpy(), 
            tf.convert_to_tensor(_kernel_powers).numpy(), 
            use_blocks = True)

        ax.plot(np.linspace(0, MAXFREQ, 1000), spectral_block_1a, label='SymBand_'+str(_), 
                linewidth=.8)
    """

    Z_np = model.inducing_variable.Z.numpy()
    ax.scatter(Z_np, np.zeros_like(Z_np),
            s=250, marker="*", alpha=0.95, c = 'cyan',
            linewidth=1, label = 'Inducing Points')


def plot_kernel(
    model, kernel, *, optimise: bool = False
) -> None:


    _, (samples_ax, prediction_parametric_ax, prediction_non_parametric_Qff_ax, 
        prediction_non_parametric_Kff_ax,
        spectrum_ax) = plt.subplots(nrows=5, ncols=1, figsize=(18, 6 * 5))
    plot_kernel_samples(samples_ax, kernel)
    plot_kernel_prediction(prediction_parametric_ax, model, optimise=optimise)
    plot_kernel_prediction(prediction_non_parametric_Kff_ax, model, optimise=optimise, parametric=False, Qff = False)
    plot_kernel_prediction(prediction_non_parametric_Qff_ax, model, optimise=optimise, parametric=False, Qff = True)
    plot_spectrum_blocks(spectrum_ax, model, data_object)

    plt.tight_layout()
    plt.savefig(f'./figures_troubleshooting/num_ind_{N_COMPONENTS}/{MODEL}_{KERNEL}_{EXPERIMENT}_num_ind_{N_COMPONENTS}_exploration.png')
    plt.close()

plot_kernel(model, kern, optimise=False)


def plot_covariance(model):
    """
    #TODO -- write documentation
    """
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

    plt.savefig(f'./figures_troubleshooting/num_ind_{N_COMPONENTS}/{MODEL}_{KERNEL}_{EXPERIMENT}_num_ind_{N_COMPONENTS}_covariances.png')
    plt.close()

plot_covariance(model)
