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
# # Inducing Frequency Bands with Local Spectrum -- limit alpha to 0 case.

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
from gpflow.init import riemann_approximate_periodogram_initial_components, rbf_spectral_density
from gpflow.kernels.initialisation_spectral_np import np_disjoint_initial_components

# for reproducibility of this notebook:
rng = np.random.RandomState(123)
tf.random.set_seed(42)

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


N = 10000  # Number of training observations

X = np.linspace(0, 5, N)  # X values
Y = func(X) + 0.2 * rng.randn(N, )  # Noisy Y values
data = (X, Y)

# %% [markdown]
# We plot the data along with the noiseless generating function:

# %%
plt.plot(X, Y, "x", alpha=0.2)
Xt = np.linspace(0, 5.0, 1000)[:, None]
Yt = func(Xt)
_ = plt.plot(Xt, Yt, c="k")


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

MAXFREQ = 10.
assert MAXFREQ < NYQUIST_FREQ, "MAXFREQ has to be lower than the Nyquist frequency"
N_COMPONENTS = 10
MAXITER = 1000

means_np, bandwidths_np, powers_np = riemann_approximate_periodogram_initial_components(X.reshape((-1,1)), 
                                                                                        Y.ravel(), 
                                                                                        [MAXFREQ], 
                                                                                        n_components=N_COMPONENTS, 
                                                                                        x_interval = [X.max() -  X.min()])

means_np = means_np.astype(np.float)
bandwidths_np = bandwidths_np.astype(np.float64)


print('means_np')
print(means_np)
print(means_np.shape)

print('bandwidths_np')
print(bandwidths_np)
print(bandwidths_np.shape)


#powers_np = np.ones(N_COMPONENTS, )

# TODO -- why is this here?
#powers_np = [np.float64(np_float) * bandwidths_np[:,_][0] * 2. for _, np_float in enumerate(powers_np)]
powers_np = [np.float64(np_float) for np_float in powers_np]

print('powers_np')
print(powers_np)

#NOTE -- alpha needs to be set to a very low value, i.e., close to 0.
ALPHA = 1e-3


kern = gpflow.kernels.MultipleSpectralBlock(n_components=N_COMPONENTS, means= means_np, 
    bandwidths= bandwidths_np, powers=powers_np, alpha=ALPHA)

fig, ax = plt.subplots(1,1, figsize=(5, 2.5))

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

plt.savefig('./figures/IFB_sgpr_sym_rectangles_init.png')
plt.close()

ind_var = gpflow.inducing_variables.RectangularSpectralInducingPoints(kern = kern)

print('-------- Inducing Variables --------')
print(ind_var)


#m = gpflow.models.SVGP(kern, gpflow.likelihoods.Gaussian(), ind_var)
m = gpflow.models.SGPR((X.reshape((-1,1)), Y.reshape((-1,1))), kern, ind_var)

print('--------------------------')
print('trainable variables at the beginning')
print(m.trainable_variables)
gpflow.utilities.set_trainable(m.kernel.bandwidths, False)
gpflow.utilities.set_trainable(m.kernel.means, False)
gpflow.utilities.set_trainable(m.kernel.powers, False)
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

#means_np = np.ones((1,N_COMPONENTS)) 
#bandwidths_np = np.ones((1,N_COMPONENTS)) 
#powers_np = np.ones(N_COMPONENTS, )

#powers_np = [np.float64(np_float) * bandwidths_np[:,_][0] * 2. for _, np_float in enumerate(powers_np)]


# %% [markdown]
# Finally, we plot the model's predictions.

def plot(title=""):
    plt.figure(figsize=(12, 4))
    plt.title(title)
    pX = np.linspace(-2.0, 7.0, 1000)[:, None]  # Test locations
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
plt.savefig('./figures/IFB_sgpr_pred_toy_data.png')
plt.close()

def plot_samples(title=""):
    plt.figure(figsize=(12, 4))
    plt.title(title)
    pX = np.linspace(-2.0, 7.0, 1000)[:, None]  # Test locations
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
plt.savefig('./figures/IFB_sgpr_samples_toy_data.png')
plt.close()

# Periodogram and optimized symmetrical rectangles

MAXFREQ=15.
ax = data_object.plot_spectrum(maxfreq=MAXFREQ)

for _ in range(N_COMPONENTS):

    """
    spectral_block_1a = make_component_spectrum(np.linspace(0, MAXFREQ, 1000), 
        tf.convert_to_tensor(kern.kernels[_].means).numpy(), 
        tf.convert_to_tensor(kern.kernels[_].bandwidths).numpy(), 
        tf.convert_to_tensor(kern.kernels[_].powers).numpy(), 
        use_blocks = True)
    """

for _ in range(N_COMPONENTS):
    spectral_block_1a = make_component_spectrum(np.linspace(0, MAXFREQ, 1000), 
                                                means_np[:,_], bandwidths_np[:,_], 
                                                powers_np[_], use_blocks = True)
    ax.plot(np.linspace(0, MAXFREQ, 1000), spectral_block_1a.ravel(), 
            label='$S_{aa}(\\nu)$', linewidth=.8)


    spectral_block_1a = make_component_spectrum(np.linspace(0, MAXFREQ, 1000), 
        tf.convert_to_tensor(kern.means[:,_]).numpy(), 
        tf.convert_to_tensor(kern.bandwidths[:,_]).numpy(), 
        tf.convert_to_tensor(kern.powers[_]).numpy(), 
        use_blocks = True)

    ax.plot(np.linspace(0, MAXFREQ, 1000), spectral_block_1a, label='SB_'+str(_), 
            linewidth=.8)
EXPERIMENT_NAME = 'periodogram_init_svgp_freq_bands'

plt.savefig('./figures/IFB_sgpr_toy_data_periodogram.png')
plt.close()
