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
# # GP-MultiSinc.

# %%
# %matplotlib inline
import itertools
import time

import pandas as pd
from sklearn.cluster import KMeans
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

# %% [markdown]
# ## Generating data
# For this notebook example, we use the Motor dataset.

def motorcycle_data():
    """
    The motorcycle dataset where the targets are normalised to zero mean and unit variance.
    Returns a tuple of input features with shape [N, 1] and corresponding targets with shape [N, 1].
    """
    df = pd.read_csv("./data/motor.csv", index_col=0)
    X, Y = df["times"].values.reshape(-1, 1), df["accel"].values.reshape(-1, 1)
    Y = (Y - Y.mean()) / Y.std()
    return X, Y


X, Y = motorcycle_data()
plt.plot(X, Y, "kx")
plt.xlabel("time")
plt.ylabel("Acceleration")
plt.savefig("./figures/motor.png")
plt.close()
#X = tf.convert_to_tensor(X, dtype=tf.float64)
#Y = tf.convert_to_tensor(Y, dtype=tf.float64)
X = X.astype(np.float64)
Y = Y.astype(np.float64)
data = (X, Y)


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

X = X.astype(np.float64)
Y = Y.astype(np.float64)
Y = Y.ravel()

print('shape of data')
print(X.shape)
print(Y.shape)

data_object = gpflow.data.Data(X = X, Y = Y)
NYQUIST_FREQ = data_object.get_nyquist_estimation()

print('**********************')
print('nyguist frequency')
print(NYQUIST_FREQ)
print('**********************')

MODEL = 'gp_multi_sinc_sgpr'
EXPERIMENT = 'motor_data'
MAXFREQ = 2.
assert MAXFREQ < NYQUIST_FREQ, "MAXFREQ has to be lower than the Nyquist frequency"
N_COMPONENTS = 50
MAXITER = 100

#INIT_METHOD = 'rbf'
#INIT_METHOD = 'Periodogram' 
INIT_METHOD ='Neutral'
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

#powers_np = [np.float64(np_float) * bandwidths_np[:,_][0] * 2. for _, np_float in enumerate(powers_np)]
powers_np = [np.float64(np_float) for np_float in powers_np]

print('powers_np')
print(powers_np)

kern = gpflow.kernels.MultipleSpectralBlock(n_components=N_COMPONENTS, means= means_np, 
    bandwidths= bandwidths_np, powers=powers_np, alpha=ALPHA)

fig, ax = plt.subplots(1, 1, 
                       figsize=(5, 2.5))

for _ in range(N_COMPONENTS):
    spectral_block_1a = make_component_spectrum(np.linspace(0, MAXFREQ, 1000), 
                                                means_np[:,_], bandwidths_np[:,_], 
                                                powers_np[_], use_blocks = True)
    ax.plot(np.linspace(0, MAXFREQ, 1000), spectral_block_1a.ravel(), 
            label='$S_{aa}(\\nu)$', linewidth=.8)

spectral_block_1a = rbf_spectral_density(np.linspace(0, MAXFREQ, 1000) 
    )
ax.plot(np.linspace(0, MAXFREQ, 1000), spectral_block_1a.ravel(), 
        label='RBF spectral density', linewidth=.8)

plt.savefig(f'./figures/{MODEL}_sym_rectangles_init_{EXPERIMENT}.png')
plt.close()


km = KMeans(n_clusters=N_COMPONENTS).fit(X.reshape((-1,1)))
Z = km.cluster_centers_
ind_var = gpflow.inducing_variables.InducingPoints(Z = Z)

print('-------- Inducing Variables --------')
print(ind_var)

#m = gpflow.models.SVGP(kern, gpflow.likelihoods.Gaussian(), ind_var)
m = gpflow.models.SGPR((X.reshape((-1,1)), Y.reshape((-1,1))), kern, ind_var)

print('--------------------------')
print('trainable variables at the beginning')
print(m.trainable_variables)
gpflow.utilities.set_trainable(m.kernel.bandwidths, False)
gpflow.utilities.set_trainable(m.kernel.means, False)
#gpflow.utilities.set_trainable(m.kernel.powers, False)
print('--------------------------')
print('trainable variables after deactivation')
print(m.trainable_variables)

opt_logs = gpflow.optimizers.Scipy().minimize(
    m.training_loss,
    variables=m.trainable_variables,
    method="l-bfgs-b",
    options={"disp": True, "maxiter": MAXITER},
)

print('---- After training -----')
print('means_np')
print(tf.convert_to_tensor(kern.means).numpy())

print('bandwidths_np')
print(tf.convert_to_tensor(kern.bandwidths).numpy())

print('powers_np')
print(tf.convert_to_tensor(kern.powers).numpy())

# %% [markdown]
# Finally, we plot the model's predictions.

def plot(title=""):
    plt.figure(figsize=(12, 4))
    plt.title(title)
    pX = np.linspace(-10.0, 55.0, 1000)[:, None]  # Test locations
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
plt.savefig(f'./figures/{MODEL}_pred_{EXPERIMENT}.png')
plt.close()

def plot_samples(title=""):
    plt.figure(figsize=(12, 4))
    plt.title(title)
    pX = np.linspace(-10.0, 55.0, 1000)[:, None]  # Test locations
    pY, pYv = m.predict_y(pX)  # Predict Y values at test locations
    #TODO -- we need to take samples actually
    
    tf.random.set_seed(43)
    n_samples = 20
    # predict_f_samples draws n_samples examples of the function f, and returns their values at Xplot.
    fs = m.predict_f_samples(pX, n_samples)
    plt.plot(pX, fs[:, :, 0].numpy().T)
    #ax.set_ylim(bottom=-2.0, top=2.0)
    #ax.set_title("Example $f$s")

    plt.plot(X, Y, "x", label="Training points", alpha=0.2)
    (line,) = plt.plot(pX, pY, lw=1.5, label="Mean of predictive posterior")
    #col = line.get_color()
    #plt.fill_between(
    #    pX[:, 0],
    #    (pY - 2 * pYv ** 0.5)[:, 0],
    #    (pY + 2 * pYv ** 0.5)[:, 0],
    #    color=col,
    #    alpha=0.6,
    #    lw=1.5,
    #)
    #Z = m.inducing_variable.Z.numpy()
    #plt.plot(Z, np.zeros_like(Z), "k|", mew=2, label="Inducing locations")
    plt.legend(loc="lower right")

plot_samples("Sample Predictions after training")
plt.savefig(f'./figures/{MODEL}_samples_{EXPERIMENT}.png')
plt.close()

# Periodogram and optimized symmetrical rectangles

MAXFREQ=15.
ax = data_object.plot_spectrum(maxfreq=MAXFREQ)

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
EXPERIMENT_NAME = '....'

plt.savefig(f'./figures/{MODEL}_{EXPERIMENT}_periodogram.png')
plt.close()
