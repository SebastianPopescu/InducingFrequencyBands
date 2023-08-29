# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np
from check_shapes import check_shapes
import math

from ..config import default_float
from ..inducing_variables import (
    InducingPatches, 
    InducingPoints, 
    Multiscale, 
    SpectralInducingVariables, 
    SymRectangularSpectralInducingPoints, 
    SymRectangularDiracDeltaSpectralInducingPoints,
)
from ..kernels import (
    Convolutional, 
    Kernel, 
    SquaredExponential, 
    MultipleSpectralBlock, 
    SpectralKernel, 
    MultipleDiracDeltaSpectralBlock,
)
from .dispatch import Kuu


@Kuu.register(InducingPoints, Kernel)
@check_shapes(
    "inducing_variable: [M, D, 1]",
    "return: [M, M]",
)
def Kuu_kernel_inducingpoints(
    inducing_variable: InducingPoints, kernel: Kernel, *, jitter: float = 0.0
) -> tf.Tensor:
    Kzz = kernel(inducing_variable.Z)
    Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)
    return Kzz

@Kuu.register(InducingPoints, MultipleSpectralBlock)
def Kuu_spectral_kernel_inducingpoints(
    inducing_variable: InducingPoints, kernel: MultipleSpectralBlock, *, jitter: float = 0.0
) -> tf.Tensor:

    """
    To be used for SGPR or SVGP versions of GP-Sinc.
    """

    Kzz = kernel(inducing_variable.Z)
    Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)

    return Kzz

@Kuu.register(SpectralInducingVariables, SpectralKernel)
def Kuu_block_spectral_kernel_inducingpoints(
    inducing_variable: SpectralInducingVariables, kernel: SpectralKernel, *, jitter: float = 0.0
) -> tf.Tensor:
    
    """
    To be used for example for GP-Sinc or GP-MultiSpectralKernel.
    """
    
    Kzz = kernel(inducing_variable.Z)
    Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)
    return Kzz


@Kuu.register(SymRectangularDiracDeltaSpectralInducingPoints, MultipleDiracDeltaSpectralBlock)
def Kuu_rectangular_dirac_delta_spectral_kernel_inducingpoints(
    inducing_variable: SpectralInducingVariables, kernel: SpectralKernel, *, jitter: float = 0.0
) -> tf.Tensor:
    
    r"""
    Local Spectrum \mathcal{F}_{c}\left(  \xi \right) 
    is a complex GP, meaning it posses both a covariance
    and a pseudo-covariance, from which we can obtain
    the covariances for its real and imaginary components.

    #TODO -- find eqns on Overleaf.
    K and P correspond to equations ...

    Real and Imaginary covariances correspond to equations ... 

    _means = kernel.means # expected shape [D, M]
    _bandwidths = kernel.bandwidths # expected shape [D, M]
    _powers = kernel.powers # expected shape [M, ]

    To be used for Inducing Frequency Bands models with Dirac Delta bandwidths for its ind. pts.
    """

    #concat_freqs = tf.concat([-kernel.means,kernel.means], axis = 1)

    # Local Spectrum covariance         
    #K = dirac_spectrum_covariance(concat_freqs, concat_freqs, kernel) # [2M, 2M]
    K = dirac_spectrum_covariance(kernel.means, kernel.means, kernel) # [M, M]
    # Local Spectrum pseudo-covariance
    #P = dirac_spectrum_covariance(concat_freqs, -concat_freqs, kernel) # [2M, 2M]
    P = dirac_spectrum_covariance(kernel.means, -kernel.means, kernel) # [M, M]

    # Krr -- real covariance
    real_cov = 0.5*(K + P) # [2M, 2M]
    # Kii -- imaginary covariance
    imag_cov = 0.5*(K - P) # [2M, 2M]
    #NOTE -- remainder: Kir = Kri = 0 since the underlying signal is real-valued.

    Kzz = BlockDiagMat(real_cov, imag_cov) # [4M, 4M]
    Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype) # [4M, 4M]

    return Kzz # [4M, 4M]

def dirac_spectrum_covariance(xi1, xi2, kernel):

    r"""
    Computes the Dirac Delta scenario spectrum covariance for local spectrum 
    \mathcal{F}_{c}(\xi), taking into account both negative and positive frequencies.
    """

    print('--- inside dirac_spectrum_covariance ---')

    diff = tf.reshape(xi1,[-1,1]) -  tf.reshape(xi2, [1,-1])
    Kzz = math.pi * tf.cast(tf.math.reciprocal(kernel.alpha), default_float()) # [1, ]
    Kzz *= tf.math.exp(- math.pi**2 * tf.cast(tf.math.reciprocal(2. * kernel.alpha), default_float()) * 
                       tf.square(diff)
                       ) # [M, M]
    print('Kzz')
    print(Kzz)

    spectrum_powers = kernel.powers * tf.cast(tf.math.reciprocal(2.), default_float()) 
    spectrum_powers = tf.reshape(spectrum_powers, [-1,1,1]) # [Q, 1, 1]
    print("spectrum_powers")
    print(spectrum_powers)


    rho = (tf.reshape(xi1,[-1,1]) + 
           tf.reshape(xi2, [1,-1])) * 0.5 # [M, M]
    rho = rho[tf.newaxis,...] # [1, M, M]
    print("rho")
    print(rho)

    exp_pos_freq = tf.math.exp(-2*math.pi**2 * tf.cast(tf.math.reciprocal(kernel.alpha), default_float()) * 
                       tf.square(rho - tf.reshape(kernel.means, [-1,1,1]))
                       ) # [Q, M, M]
    print("exp_pos_freq")
    print(exp_pos_freq)

    exp_neg_freq = tf.math.exp(-2*math.pi**2 * tf.cast(tf.math.reciprocal(kernel.alpha), default_float()) * 
                       tf.square(rho + tf.reshape(kernel.means, [-1,1,1]))
                       ) # [Q, M, M]
    print(exp_neg_freq)
    exp_part = spectrum_powers * (exp_pos_freq + exp_neg_freq) # [Q, M, M]
    print("exp_part")
    print(exp_part)
    Kzz *= tf.reduce_sum(exp_part, axis = 0) # [M, M]
    print("Kzz")
    print(Kzz)

    return Kzz


### Helper functions ###

def outersum(a, b):
    _ = tf.experimental.numpy.outer(a, tf.ones_like(b))
    __ = tf.experimental.numpy.outer(tf.ones_like(a), b)

    return _ + __

def BlockDiagMat(A, B):

    tl_shape = tf.stack([A.shape[0], B.shape[1]])
    br_shape = tf.stack([B.shape[0], A.shape[1]])
    top = tf.concat([A, tf.zeros(tl_shape, default_float())], axis=1)
    bottom = tf.concat([tf.zeros(br_shape, default_float()), B], axis=1)

    return tf.concat([top, bottom], axis=0)




@Kuu.register(Multiscale, SquaredExponential)
@check_shapes(
    "inducing_variable: [M, D, 1]",
    "return: [M, M]",
)
def Kuu_sqexp_multiscale(
    inducing_variable: Multiscale, kernel: SquaredExponential, *, jitter: float = 0.0
) -> tf.Tensor:
    Zmu, Zlen = kernel.slice(inducing_variable.Z, inducing_variable.scales)
    idlengthscales2 = tf.square(kernel.lengthscales + Zlen)
    sc = tf.sqrt(
        idlengthscales2[None, ...] + idlengthscales2[:, None, ...] - kernel.lengthscales ** 2
    )
    d = inducing_variable._cust_square_dist(Zmu, Zmu, sc)
    Kzz = kernel.variance * tf.exp(-d / 2) * tf.reduce_prod(kernel.lengthscales / sc, 2)
    Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)
    return Kzz

@Kuu.register(InducingPatches, Convolutional)
@check_shapes(
    "inducing_variable: [M, D, 1]",
    "return: [M, M]",
)
def Kuu_conv_patch(
    inducing_variable: InducingPatches, kernel: Convolutional, jitter: float = 0.0
) -> tf.Tensor:
    return kernel.base_kernel.K(inducing_variable.Z) + jitter * tf.eye(
        inducing_variable.num_inducing, dtype=default_float()
    )
