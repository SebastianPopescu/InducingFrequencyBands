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
# # Variational Fourier Features --demo of Kuf in abstract case.

# %%
# %matplotlib inline
import itertools
import time

import numpy.random as rnd
import scipy.stats as stats
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
from gpflow import covariances
from gpflow.inducing_variables import SpectralInducingPoints
from gpflow.init import riemann_approximate_periodogram_initial_components, rbf_spectral_density, neutral_initial_components, riemann_approximate_rbf_initial_components
from gpflow.kernels.initialisation_spectral_np import np_disjoint_initial_components


# for reproducibility of this notebook:
rng = np.random.RandomState(123)
tf.random.set_seed(42)

# %% [markdown]
# In this notebook we adapt Figure 2 from VFF paper to different scenarios 
# of the Inducing Frequency Bands.


a = -10.
b = 10.
N = 1000  # Number of training observations
MARGIN = 0.5
X = np.linspace(a-MARGIN, b+MARGIN, N)  # X values to evaluate Kuf at
X = X.astype(np.float64)

X = X.astype(np.float64)
Y = X.astype(np.float64) # dummy 
Y = Y.ravel()

data_object = gpflow.data.Data(X = X, Y = Y)
NYQUIST_FREQ = data_object.get_nyquist_estimation()

print('**********************')
print('nyguist frequency')
print(NYQUIST_FREQ)
print('**********************')
X = X.reshape((-1,1))

print('shape of data')
print(X.shape)

EXPERIMENT = 'Kuf'

#BLOCKS = 'symmetrical_rec'
BLOCKS = 'asymmetrical_rec'
MODEL = 'ifb_sgpr'
MAXFREQ = 10.
assert MAXFREQ < NYQUIST_FREQ, "MAXFREQ has to be lower than the Nyquist frequency"
N_COMPONENTS = 10
MAXITER = 1000

INIT_METHOD = 'rbf'
#INIT_METHOD = 'Periodogram' 
#INIT_METHOD ='Neutral'
DELTAS = 1e-1
#NOTE -- alpha needs to be set to a very low value, i.e., close to 0.
ALPHA = 1e-12

if INIT_METHOD == 'Periodogram':

    means_np, bandwidths_np, powers_np = riemann_approximate_periodogram_initial_components(
        X.reshape((-1,1)), 
        Y.ravel(), 
        [MAXFREQ], 
        n_components=N_COMPONENTS, 
        x_interval = [X.max() -  X.min()]
        )
elif INIT_METHOD == 'Neutral':

    means_np, bandwidths_np, powers_np = neutral_initial_components(
        X.reshape((-1,1)), 
        Y.ravel(), 
        [MAXFREQ], 
        n_components=N_COMPONENTS,
        x_interval = [X.max() -  X.min()],
        deltas = DELTAS,                                                                   
        )
elif INIT_METHOD == 'rbf':
    
    means_np, bandwidths_np, powers_np = riemann_approximate_rbf_initial_components(
        [MAXFREQ], 
        n_components=N_COMPONENTS, 
        x_interval = [X.max() -  X.min()]
        )

means_np = means_np.astype(np.float)
bandwidths_np = bandwidths_np.astype(np.float64)

print('means_np')
print(means_np)
print(means_np.shape)

print('bandwidths_np')
print(bandwidths_np)
print(bandwidths_np.shape)

powers_np = [np.float64(np_float) for np_float in powers_np]

kern = gpflow.kernels.DecomposedMultipleSpectralBlock(n_components=N_COMPONENTS, 
                                                          means= means_np, 
                                                          bandwidths= bandwidths_np, 
                                                          real_powers= powers_np, 
                                                          img_powers = powers_np,
                                                          alpha=ALPHA)
ind_pts = gpflow.inducing_variables.AsymRectangularSpectralInducingPoints(kern = kern)

#FIXME -- it can't find the signature 
#_Kuf = covariances.Kuf(inducing_variable = ind_pts, kernel = kern, Xnew = X)
#print(_Kuf)

def Kuf_sym_block_spectral_kernel_inducingpoints(
    inducing_variable, kernel, Xnew):

    _means = kernel.means # expected shape [D, M]
    _bandwidths = kernel.bandwidths # expected shape [D, M]
    _powers = kernel.powers # expected shape [M, ]

    sine_term = tf.reduce_prod( tf.sin(0.5 * tf.multiply(tf.transpose(_bandwidths)[..., None], # [M, D, 1]
        tf.transpose(Xnew)[None, ...] # [1, D, N]
    ) #[M, D, N]
    ), axis = 1) #[M, N]
    
    cosine_term = tf.reduce_prod( tf.cos( tf.multiply(tf.transpose(_means)[..., None], # [M, D, 1]
        tf.transpose(Xnew)[None, ...] # [1, D, N]
    ) #[M, D, N]
    ), axis = 1) #[M, N]

    pre_multiplier = 2. * _powers * tf.reduce_prod(tf.math.reciprocal(_bandwidths), axis = 0) # expected shape (M, )

    Kzf  = pre_multiplier[..., None] * sine_term * cosine_term # expected shape (M, N)

    return Kzf


def Kuf_asym_block_spectral_kernel_inducingpoints(
    inducing_variable, kernel, Xnew):

    _means = kernel.means # expected shape [D, M]
    _bandwidths = kernel.bandwidths # expected shape [D, M]
    _powers = kernel.powers # expected shape [M, ]

    #real part

    r_sine_term = tf.reduce_prod( tf.sin(0.5 * tf.multiply(tf.transpose(_bandwidths)[..., None], # [M, D, 1]
        tf.transpose(Xnew)[None, ...] # [1, D, N]
    ) #[M, D, N]
    ), axis = 1) #[M, N]
    
    r_cosine_term = tf.reduce_prod( tf.cos( tf.multiply(tf.transpose(_means)[..., None], # [M, D, 1]
        tf.transpose(Xnew)[None, ...] # [1, D, N]
    ) #[M, D, N]
    ), axis = 1) #[M, N]

    r_pre_multiplier = _powers * tf.reduce_prod(tf.math.reciprocal(_bandwidths), axis = 0) # expected shape (M, )

    real_part  = r_pre_multiplier[..., None] * r_sine_term * r_cosine_term # expected shape (M, N)

    i_sine_term = tf.reduce_prod( tf.sin( tf.multiply(tf.transpose(_means)[..., None], # [M, D, 1]
        tf.transpose(Xnew)[None, ...] # [1, D, N]
    ) #[M, D, N]
    ), axis = 1) #[M, N]
    i_pre_multiplier = - _powers * tf.reduce_prod(tf.math.reciprocal(_bandwidths), axis = 0) # expected shape (M, )

    img_part  = i_pre_multiplier[..., None] * r_sine_term * i_sine_term # expected shape (M, N)

    return real_part, img_part


if BLOCKS == 'symmetrical_rec':
    _Kuf = Kuf_sym_block_spectral_kernel_inducingpoints(ind_pts, kern, X)
    Kuf_np = _Kuf.numpy()

    fig, ax = plt.subplots(1, 1, 
                        figsize=(5, 2.5))
    for _ in range(N_COMPONENTS):
        ax.plot(X, Kuf_np[_,:].ravel(), 
                label=f'Kuf SymRec{_}', linewidth=.4)
    plt.plot(X, stats.norm.pdf(X, 0., ALPHA), color = 'darkorange', linewidth=1.)
    #ax.legend()
    plt.savefig(f'./figures/{MODEL}_{BLOCKS}_{EXPERIMENT}_{INIT_METHOD}.png')
    plt.close()

elif BLOCKS =='asymmetrical_rec':
    _Kuf_real, _Kuf_img = Kuf_asym_block_spectral_kernel_inducingpoints(ind_pts, kern, X)
    Kuf_real_np = _Kuf_real.numpy()
    Kuf_img_np = _Kuf_img.numpy()


    fig, ax = plt.subplots(2, 1, 
                        figsize=(5, 2.5 * 2))
    for _ in range(N_COMPONENTS):
        ax[0].plot(X, Kuf_real_np[_,:].ravel(), 
                label=f'Real Kuf AsymRec{_}', linewidth=.4)
    ax[0].set_title('Real features')
    ax[0].plot(X, stats.norm.pdf(X, 0., ALPHA), color = 'darkorange', linewidth=1.)

    for _ in range(N_COMPONENTS):
        ax[1].plot(X, Kuf_img_np[_,:].ravel(), 
                label=f'Img Kuf AsymRec{_}', linewidth=.4)
    ax[1].set_title('Img features')
    ax[1].plot(X, stats.norm.pdf(X, 0., ALPHA), color = 'darkorange', linewidth=1.)
    #ax.legend()
    plt.savefig(f'./figures/{MODEL}_{BLOCKS}_{EXPERIMENT}_{INIT_METHOD}.png')
    plt.close()

