import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import gpflow
from gpflow.ci_utils import reduce_in_tests

plt.style.use("ggplot")

# for reproducibility of this notebook:
rng = np.random.RandomState(123)
tf.random.set_seed(42)




def spectral_basis(x, mean, bandwidth, variance, use_blocks=True):
    if use_blocks:
        spectrum = np.where(abs(x - mean) <= 0.5 * bandwidth, variance/ (2. * bandwidth), 0.)
    else:
        spectrum = norm.pdf(x, mean, bandwidth)
    return variance * spectrum


def make_component_spectrum(x, mean, bandwidth, variance, use_blocks=True):
    spectrum = 1. * (spectral_basis(x, mean, bandwidth, variance, use_blocks) + spectral_basis(x, -mean, bandwidth, variance, use_blocks))
    return spectrum

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


plt.plot(X, Y, "x", alpha=0.2)
Xt = np.linspace(-1.1, 1.1, 1000)[:, None]
Yt = func(Xt)
_ = plt.plot(Xt, Yt, c="k")


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

data_object = gpflow.data.Data(X = X, Y = Y)
NYQUIST_FREQ = data_object.get_nyquist_estimation()

print('**********************')
print(NYQUIST_FREQ)
print('**********************')

MAXFREQ=10.
N_COMPONENTS = 1
MAXITER = 1000
EXPERIMENT_NAME = 'samples_from_delta_freq'
means_np, bandwidths_np, powers_np = np_disjoint_initial_components([10.], n_components=N_COMPONENTS, x_interval = [X.max() -  X.min()])

means_np = means_np.astype(np.float64)
bandwidths_np = bandwidths_np.astype(np.float64)

means_np = np.ones_like(means_np) * 10.
bandwidths_np = np.ones_like(bandwidths_np) * 1e-12

powers_np = [np.float64(np_float) for np_float in powers_np]
print('powers_np')
print(powers_np)

kern = gpflow.kernels.SpectralDiracDeltaBlock(means= means_np, 
    bandwidths= bandwidths_np, powers=powers_np)

fig, ax = plt.subplots(1,1, figsize=(5, 2.5))

for _ in range(N_COMPONENTS):

    spectral_block_1a = make_component_spectrum(np.linspace(0, MAXFREQ, 1000), means_np[:,_], bandwidths_np[:,_], powers_np[_], use_blocks = True)
    ax.plot(np.linspace(0, MAXFREQ, 1000), spectral_block_1a.ravel(), label='$S_{aa}(\\nu)$', linewidth=.8)

plt.savefig(f'./figures/{EXPERIMENT_NAME}_init.png')
plt.close()


#ind_var = gpflow.inducing_variables.RectangularSpectralInducingPoints(kern = kern)

#Z = X[:M, :].copy() # Initialise inducing locations to the first M inputs in the dataset
#m = gpflow.models.SVGP(kern, gpflow.likelihoods.Gaussian(), ind_var)
m = gpflow.models.GPR( data = (X, Y), kernel = kern)


def plot(title=""):
    plt.figure(figsize=(12, 4))
    plt.title(title)
    pX = np.linspace(-1, 1, 100)[:, None]  # Test locations
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


plot("Predictions after training")
plt.savefig(f'./figures/{EXPERIMENT_NAME}.png')
plt.close()


# Periodogram and optimized symmetrical rectangles

MAXFREQ=15.
ax = data_object.plot_spectrum(maxfreq=MAXFREQ)

for _ in range(N_COMPONENTS):

    """
    spectral_block_1a = make_component_spectrum(np.linspace(0, MAXFREQ, 1000), 
        tf.convert_to_tensor(kern.kernels[_].means).numpy(), 
        tf.convert_to_tensor(kern.kernels[_].bandwidths).numpy()
        tf.convert_to_tensor(kern.kernels[_].powers).numpy(), 
        , use_blocks = True)
    """
    spectral_block_1a = make_component_spectrum(np.linspace(0, MAXFREQ, 1000), 
        tf.convert_to_tensor(kern.means[:,_]).numpy(), 
        tf.convert_to_tensor(kern.bandwidths[:,_]).numpy(),
        tf.convert_to_tensor(kern.powers[_]).numpy(), 
        use_blocks = True)

    ax.plot(np.linspace(0, MAXFREQ, 1000), spectral_block_1a, label='SB_'+str(_), linewidth=.8)

plt.savefig(f'./figures/{EXPERIMENT_NAME}_periodogram.png')
plt.close()
