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


plt.style.use("ggplot")

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

X = rng.rand(N, 1) * 2 - 1  # X values
Y = func(X) + 0.2 * rng.randn(N, 1)  # Noisy Y values
data = (X, Y)

# %% [markdown]
# We plot the data along with the noiseless generating function:

# %%
plt.plot(X, Y, "x", alpha=0.2)
Xt = np.linspace(-1.1, 1.1, 1000)[:, None]
Yt = func(Xt)
_ = plt.plot(Xt, Yt, c="k")

# %% [markdown]
# ## Building the modelimport numpy as np
import numpy.random as rnd
import time# ---
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

plt.style.use("ggplot")



def spectral_basis(x, mean, bandwidth, variance, use_blocks=True):
    if use_blocks:
        spectrum = np.where(abs(x - mean) <= 0.5 * bandwidth, 1./bandwidth, 0.)
    else:
        spectrum = norm.pdf(x, mean, bandwidth)
    return variance * spectrum


def make_component_spectrum(x, mean, bandwidth, variance, use_blocks=True):
    spectrum = 1. * (spectral_basis(x, mean, bandwidth, variance, use_blocks) + spectral_basis(x, -mean, bandwidth, variance, use_blocks))
    return spectrum

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

X = rng.rand(N, 1) * 2 - 1  # X values
Y = func(X) + 0.2 * rng.randn(N, 1)  # Noisy Y values
data = (X, Y)

# %% [markdown]
# We plot the data along with the noiseless generating function:

# %%
plt.plot(X, Y, "x", alpha=0.2)
Xt = np.linspace(-1.1, 1.1, 1000)[:, None]
Yt = func(Xt)
_ = plt.plot(Xt, Yt, c="k")

# %% [markdown]
# ## Building the modelimport numpy as np
import numpy.random as rnd
import time
import gpflow
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def func(x):
    return np.sin(x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)

N = 500 # Number of training observations

X = rnd.rand(N, 1) * 2 - 1 # X values
Y = func(X) + 0.2 * rnd.randn(N, 1) # Noisy Y values


X = X.astype(np.float64)
Y = Y.astype(np.float64)


plt.plot(X, Y, 'x', alpha=0.2)
D = X.shape[1]
Xt = np.linspace(-1.1, 1.1, 1000)[:, None]
Yt = func(Xt)
plt.plot(Xt, Yt, c='k');


def find_min_distance(lst):
    sorted_lst = sorted(set(lst))
    return min(n2 - n1 for n1, n2 in zip(sorted_lst, sorted_lst[1:]))

#kern = gpflow.kernels.RBF(D)
#TODO -- need to initialize the inducing variable

from gpflow.kernels.initialisation_spectral_np import np_disjoint_initial_components

X_list = X.ravel().tolist()
NYQUIST_FREQ = 1. / (2. * find_min_distance(X_list))


data_object = gpflow.data.Data(X = X, Y = Y)
NYQUIST_FREQ = data_object.get_nyquist_estimation()

print('**********************')
print(NYQUIST_FREQ)
print('**********************')


#means_np, bandwidths_np, powers_np = np_disjoint_initial_components([NYQUIST_FREQ], n_components=M, x_interval = [X.max() -  X.min()])

#means_np = means_np.astype(np.float64)
#bandwidths_np = bandwidths_np.astype(np.float64)

means_np = np.ones((1,1))
bandwidths_np = np.ones((1,1))
powers_np = [1.0]


powers_np = [np.float64(np_float) for np_float in powers_np]
print('powers_np')
print(powers_np)

#TODO -- need to change the kernel here
kern = gpflow.kernels.SpectralBlock(means= means_np, 
    bandwidths= bandwidths_np, powers=powers_np)

#ind_var = gpflow.inducing_variables.RectangularSpectralInducingPoints(kern = kern)

#Z = X[:M, :].copy() # Initialise inducing locations to the first M inputs in the dataset
#m = gpflow.models.SVGP(kern, gpflow.likelihoods.Gaussian(), ind_var)

m = gpflow.models.GPR( data = (X, Y), kernel = kern)

MAXITER = 1000
#MAXFREQ= NYQUIST_FREQ[0]
MAXFREQ = 10.


spectral_block_1a = make_component_spectrum(np.linspace(0, MAXFREQ, 1000), tf.convert_to_tensor(kern.means).numpy(), tf.convert_to_tensor(kern.bandwidths).numpy(), tf.convert_to_tensor(kern.powers).numpy(), use_blocks = True)


fig, ax = plt.subplots(1,1, figsize=(5, 2.5))
ax.plot(np.linspace(0, MAXFREQ, 1000), spectral_block_1a.ravel(), label='$S_{aa}(\\nu)$', linewidth=.8)

plt.savefig('./figures/gp_sinc_sym_rec_init.png')
plt.close()


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
    pX = np.linspace(-1, 1, 100)[:, None]  # Test locations
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
plt.savefig('./figures/gp_sinc_toy_data.png')
plt.close()




#Let's create the periogodram now


ax = data_object.plot_spectrum(maxfreq=MAXFREQ)

spectral_block_1a = make_component_spectrum(np.linspace(0, MAXFREQ, 1000), tf.convert_to_tensor(kern.means).numpy(), tf.convert_to_tensor(kern.bandwidths).numpy(), tf.convert_to_tensor(kern.powers).numpy(), use_blocks = True)

ax.plot(np.linspace(0, MAXFREQ, 1000), spectral_block_1a.ravel(), label='$S_{aa}(\\nu)$', linewidth=.8)

plt.savefig('./figures/gp_sinc_toy_data_periodogram.png')
plt.close()
