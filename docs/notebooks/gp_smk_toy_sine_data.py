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
# # Stochastic Variational Inference for scalability with SVGP

# %% [markdown]
# One of the main criticisms of Gaussian processes is their scalability to large datasets. In this notebook, we illustrate how to use the state-of-the-art Stochastic Variational Gaussian Process (SVGP) (*Hensman, et. al. 2013*) to overcome this problem.

# %%
# %matplotlib inline
import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import gpflow
from gpflow.ci_utils import reduce_in_tests
import tikzplotlib
from scipy.stats import norm

import matplotlib.pyplot as plt
plt.ion()
plt.close('all')

from gpflow.kernels.initialisation_spectral_np import np_disjoint_initial_components


plt.style.use("ggplot")

# for reproducibility of this notebook:
rng = np.random.RandomState(123)
tf.random.set_seed(42)

def spectral_basis(x, mean, bandwidth, variance, use_blocks=True):
    if use_blocks:
        spectrum = np.where(abs(x - mean) <= 0.5 * bandwidth, 1./bandwidth, 0.)
    else:
        spectrum = norm.pdf(x, mean, bandwidth)
    return variance * spectrum


def make_component_spectrum(x, mean, bandwidth, variance, use_blocks=True):
    spectrum = 1. * (spectral_basis(x, mean, bandwidth, variance, use_blocks) + spectral_basis(x, -mean, bandwidth, variance, use_blocks))
    return spectrum


# %% [markdown]
# ## Generating data
# For this notebook example, we generate 10,000 noisy observations from a test function:
# \begin{equation}
# f(x) = \sin(3\pi x) + 0.3\cos(9\pi x) + \frac{\sin(7 \pi x)}{2}
# \end{equation}

# %%
def func(x):
    return (
        np.sin(x * 3 * 3.14)
        + 0.3 * np.cos(x * 9 * 3.14)
        + 0.5 * np.sin(x * 7 * 3.14)
    )


N = 500  # Number of training observations

X = rng.rand(N, 1) * 2 - 1  # X values
Y = func(X) + 0.2 * rng.randn(N, 1)  # Noisy Y values
data = (X, Y)

# %% [markdown]
# We plot the data along with the noiseless generating function:

# %%
plt.plot(X, Y, "x", alpha=0.2)
Xt = np.linspace(-3.5, 3.5, 200)[:, None]
Yt = func(Xt)
_ = plt.plot(Xt, Yt, c="k")



data_object = gpflow.data.Data(X = X, Y = Y)
NYQUIST_FREQ = data_object.get_nyquist_estimation()

print('**********************')
print(NYQUIST_FREQ)
print('**********************')

MAXFREQ=10.
N_COMPONENTS = 10
MAXITER = 100

means_np, bandwidths_np, powers_np = np_disjoint_initial_components([10.], n_components=N_COMPONENTS, x_interval = [X.max() -  X.min()])

means_np = means_np.astype(np.float64)
bandwidths_np = bandwidths_np.astype(np.float64)

print('means_np')
print(means_np)
print(means_np.shape)

print('bandwidths_np')
print(bandwidths_np)
print(bandwidths_np.shape)

#means_np = np.ones((1,N_COMPONENTS)) 
#bandwidths_np = np.ones((1,N_COMPONENTS)) 
#powers_np = np.ones(N_COMPONENTS, )

powers_np = [np.float64(np_float) for np_float in powers_np]
print('powers_np')
print(powers_np)

#kern = gpflow.kernels.MultipleSpectralBlock(n_components=N_COMPONENTS, means= means_np, 
#    bandwidths= bandwidths_np, powers=powers_np)
kern = gpflow.kernels.MixtureSpectralGaussian(n_components= N_COMPONENTS , 
                                              powers = powers_np, 
                                              means = means_np, 
                                              bandwidths = bandwidths_np)

fig, ax = plt.subplots(1,1, figsize=(5, 2.5))

for _ in range(N_COMPONENTS):

    spectral_block_1a = make_component_spectrum(np.linspace(0, MAXFREQ, 1000), 
                                                means_np[:,_], bandwidths_np[:,_], 
                                                powers_np[_], 
                                                use_blocks = False)
    ax.plot(np.linspace(0, MAXFREQ, 1000), spectral_block_1a.ravel(), 
            label='$S_{aa}(\\nu)$', linewidth=.8)

plt.savefig('./figures/gp_smk_init.png')
plt.close()


m = gpflow.models.GPR( data = (X, Y), kernel = kern)

opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(
    m.training_loss, m.trainable_variables, options=dict(maxiter=MAXITER)
)


# %% [markdown]
# Finally, we plot the model's predictions.

# %%

# %%
def plot(title=""):
    plt.figure(figsize=(12, 4))
    plt.title(title)
    pX = np.linspace(-3.5, 3.5, 200)[:, None]  # Test locations
    pY, pYv = m.predict_y(pX)  # Predict Y values at test locations
    plt.plot(X, Y, "x", label="Training points", alpha=0.2)
    (line,) = plt.plot(pX, pY, lw=1.5, label="Mean of predictive posterior")
    col = line.get_color()
    plt.fill_between(
        pX[:, 0],
        (pY - 2 * pYv ** 0.5)[:, 0],
        (pY + 2 * pYv ** 0.5)[:, 0],
        color=col,
        alpha=0.6,
        lw=1.5,
    )

    plt.legend(loc="lower right")


plot("Predictions after training")
plt.savefig('./figures/gp_smk_toy_data.png')
plt.close()


# Periodogram and optimized symmetrical rectangles


MAXFREQ=50.
ax = data_object.plot_spectrum(maxfreq=MAXFREQ)


for _ in range(N_COMPONENTS):

    spectral_block_1a = make_component_spectrum(np.linspace(0, MAXFREQ, 1000), 
        tf.convert_to_tensor(kern.kernels[_].means).numpy().ravel(), 
        tf.convert_to_tensor(kern.kernels[_].bandwidths).numpy().ravel(), 
        tf.convert_to_tensor(kern.kernels[_].powers).numpy().ravel(), 
        use_blocks = False)


    print(tf.convert_to_tensor(kern.kernels[_].means).numpy())
    print(tf.convert_to_tensor(kern.kernels[_].powers).numpy())
    print(tf.convert_to_tensor(kern.kernels[_].bandwidths).numpy())

    ax.plot(np.linspace(0, MAXFREQ, 1000), spectral_block_1a.ravel(), label='SB_'+str(_), linewidth=.8)

plt.savefig('./figures/gp_smk_toy_data_periodogram.png')
plt.close()
