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


# for reproducibility of this notebook:
rng = np.random.RandomState(123)
tf.random.set_seed(42)

# %% [markdown]
# In this notebook we use the same hyperparameters as in Figure 2 of VFF paper.
# ωm = 16π/(b − a), and a lengthscale of l = (b − a)/10.

a = -10.
b = 10.
N = 1000  # Number of training observations
MARGIN = 0.5
X = np.linspace(a-MARGIN, b+MARGIN, N)  # X values to evaluate Kuf at
X = X.astype(np.float64)
X = X.reshape((-1,1))

print('shape of data')
print(X.shape)

MODEL = 'VFF'
EXPERIMENT = 'Kuf_demo'
N_COMPONENTS = 1

#NOTE -- this is taken to match the configuration in Figure 2 of VFF paper.
ms = 8.
#M = 10 # Number of spectral inducing points
#ms = np.arange(M) # this should be the usual ms for general experiments
omegas = (2.0 * np.pi * ms) / (b - a)

kern = gpflow.kernels.Matern12(variance = 1.0, lengthscales= (b - a)/10.)
ind_pts = SpectralInducingPoints(a = a, b = b, omegas = omegas)

#FIXME -- it can't find the signature 
#_Kuf = covariances.Kuf(inducing_variable = ind_pts, kernel = kern, Xnew = X)
#print(_Kuf)

def Kuf_L2_features_spectral_kernel_inducingpoints(
    inducing_variable, kernel, Xnew):

    lamb = tf.math.reciprocal(kernel.lengthscales)
    
    omegas = inducing_variable.omegas # shape - [M, ]   
    spectrum = inducing_variable.spectrum(kernel) # shape - [M, ]
    print('--- size of spectrum ----')
    print(spectrum) 
    a = inducing_variable.a 
    b = inducing_variable.b 
    
    #NOTE -- corresponds to real part of equation 46 from VFF paper.
    real_part = spectrum * tf.math.cos(omegas * (Xnew - a))
    real_part += spectrum * 0.5 * ( -tf.math.exp(a - Xnew) - tf.math.exp(Xnew - b))
    #NOTE -- corresponds to imaginary part of equation 46 from VFF paper.
    imaginary_part = spectrum * tf.math.sin(omegas * (Xnew - a))
    imaginary_part += spectrum  * tf.math.reciprocal(2. * lamb) * omegas * (tf.math.exp(a - Xnew) 
                                                                              - tf.math.exp(Xnew - b))

    return real_part, imaginary_part

real_Kuf, im_Kuf = Kuf_L2_features_spectral_kernel_inducingpoints(ind_pts, kern, X)

real_Kuf_np = real_Kuf.numpy()
im_Kuf_np = im_Kuf.numpy()
fig, ax = plt.subplots(1, 1, 
                       figsize=(5, 2.5))

ax.plot(X, real_Kuf_np.ravel(), 
        label='real Kuf Matern1/2', linewidth=.8, color = 'red')
ax.plot(X, im_Kuf_np.ravel(), 
        label='imaginary Kuf Matern1/2', linewidth=.8, color = 'blue')
plt.axvline(x=a, color='grey')
plt.axvline(x=b, color='grey')
ax.legend()
plt.savefig(f'./figures/{MODEL}_{EXPERIMENT}.png')
plt.close()






