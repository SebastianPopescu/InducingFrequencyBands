import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import gpflow
from gpflow.ci_utils import reduce_in_tests


from gpflow.init import (rbf_spectral_density,
    matern_1_2_spectral_density,
    matern_3_2_spectral_density, 
    matern_5_2_spectral_density
)

plt.style.use("ggplot")
    
MAXFREQ=10.
N_COMPONENTS = 1
MAXITER = 1000
EXPERIMENT_NAME = 'rbf_spectral_density'


fig, ax = plt.subplots(1,1, figsize=(5, 2.5))

for _ in range(N_COMPONENTS):

    spectral_block_1a = rbf_spectral_density(np.linspace(0, MAXFREQ, 1000), 
        0.301, 
        1.0, 
        )
    ax.plot(np.linspace(0, MAXFREQ, 1000), spectral_block_1a.ravel(), label='$S_{aa}(\\nu)$', linewidth=.8)

plt.savefig(f'./figures/{EXPERIMENT_NAME}_init.png')
plt.close()


EXPERIMENT_NAME = 'matern_1_2_spectral_density'


fig, ax = plt.subplots(1,1, figsize=(5, 2.5))

for _ in range(N_COMPONENTS):

    spectral_block_1a = matern_1_2_spectral_density(np.linspace(0, MAXFREQ, 1000), 
        0.301, 
        1.0, 
        )
    ax.plot(np.linspace(0, MAXFREQ, 1000), spectral_block_1a.ravel(), label='$S_{aa}(\\nu)$', linewidth=.8)

plt.savefig(f'./figures/{EXPERIMENT_NAME}_init.png')
plt.close()



EXPERIMENT_NAME = 'matern_3_2_spectral_density'


fig, ax = plt.subplots(1,1, figsize=(5, 2.5))

for _ in range(N_COMPONENTS):

    spectral_block_1a = matern_3_2_spectral_density(np.linspace(0, MAXFREQ, 1000), 
        0.301, 
        1.0, 
        )
    ax.plot(np.linspace(0, MAXFREQ, 1000), spectral_block_1a.ravel(), label='$S_{aa}(\\nu)$', linewidth=.8)

plt.savefig(f'./figures/{EXPERIMENT_NAME}_init.png')
plt.close()



EXPERIMENT_NAME = 'matern_5_2_spectral_density'


fig, ax = plt.subplots(1,1, figsize=(5, 2.5))

for _ in range(N_COMPONENTS):

    spectral_block_1a = matern_5_2_spectral_density(np.linspace(0, MAXFREQ, 1000), 
        0.301, 
        1.0, 
        )
    ax.plot(np.linspace(0, MAXFREQ, 1000), spectral_block_1a.ravel(), label='$S_{aa}(\\nu)$', linewidth=.8)

plt.savefig(f'./figures/{EXPERIMENT_NAME}_init.png')
plt.close()

