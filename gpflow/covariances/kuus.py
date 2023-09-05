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
    IFFRectangularSpectralInducingPoints,
) 
from ..kernels import (
    Convolutional, 
    Kernel, 
    SquaredExponential, 
    MultipleSpectralBlock, 
    SpectralKernel, 
    MixtureSpectralGaussianVectorized
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

#my own version
def rbf_spectral_density(freq, lengthscale, variance):

    _pi = math.pi

    constant_num = tf.sqrt(tf.sqrt(lengthscale)) / tf.cast((2. * tf.sqrt(_pi)), default_float())
    freq_term = tf.exp(- tf.sqrt(lengthscale) * freq**2 * 0.25)
    S = constant_num * freq_term

    return S * variance

@Kuu.register(IFFRectangularSpectralInducingPoints, SquaredExponential)
def Kuu_IFF_inducingpoints(
    inducing_variable: IFFRectangularSpectralInducingPoints, 
    kernel: SquaredExponential, *, jitter: float = 0.0
) -> tf.Tensor:
    
    r"""
    To be used for IFF. In the paper, they only use SqExp kernels.
    
    When rescaled as Kuf, so that Kuf is independent of the hyperparameters, 
    Kuu is just a diagonal matrix of 
    spectral density evaluations scaled by 1/(ε_1...ε_D)
    """
    
    #Talay's codebase
    #vol = lab.prod(inducing_variable.epsilon)
    #return tf.linalg.LinearOperatorInversion(tf.linalg.LinearOperatorDiag(
    #            (spectral_density(kernel, inducing_variable.z)*vol),
    #            is_non_singular=True,
    #            is_self_adjoint=True,
    #            is_positive_definite=True,
    #            is_square=True
    #        ))

    _means = inducing_variable.Z # expected shape [D, M]
    _lengthscales = kernel.lengthscales # [D,]
    _variance = kernel.variance # [D,]
 
    #TODO -- this squeeze is a hack, fix underlying issue
    spectral_component = tf.squeeze(
        rbf_spectral_density(freq = _means, lengthscale = _lengthscales, 
                             variance = _variance), 
                             axis = 0) #  [M, ]

    #spectral_component = spectral_density_gaussian(kernel, inducing_variable.Z)

    Kzz = tf.linalg.diag(tf.math.reciprocal(
        tf.concat([spectral_component, spectral_component], axis=0)
        ))

    Kzz*= tf.cast(tf.math.reciprocal(inducing_variable.epsilon), default_float())
    Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)

    return Kzz


@Kuu.register(SymRectangularSpectralInducingPoints, MixtureSpectralGaussianVectorized)
def Kuu_windowed_mixture_gaussian_spectral_kernel_inducingpoints(
    inducing_variable: SymRectangularSpectralInducingPoints, kernel: MixtureSpectralGaussianVectorized, *, jitter: float = 0.0
) -> tf.Tensor:
    
    """
    To be used for ReverseBNSE.
    """

    # Spectrum covariance
    # 0.5 scaling is due to missing 0.5 scaling in ``spectrum_covariance''          
    K = 0.5*(spectrum_covariance(inducing_variable.Z, inducing_variable.Z, kernel.means, kernel) + 
            spectrum_covariance(inducing_variable.Z, inducing_variable.Z, -kernel.means, kernel))
    # Spectrum pseudo-covariance
    P = 0.5*(spectrum_covariance(inducing_variable.Z, -inducing_variable.Z, kernel.means, kernel) + 
            spectrum_covariance(inducing_variable.Z, -inducing_variable.Z, -kernel.means, kernel))
    # Krr -- real covariance
    #TODO -- maybe add jitter later on
    real_cov = 0.5*(K + P) #+ 1e-8*tf.eye(N, dtype = default_float())
    # Kii -- imaginary covariance
    #TODO -- maybe add jitter later on
    imag_cov = 0.5*(K - P) #+ 1e-8*tf.eye(N, dtype = default_float())
    #NOTE -- remainder: Kir = Kri = 0 since the underlying signal is real-valued.

    return BlockDiagMat(real_cov, imag_cov)

def spectrum_covariance(xi1, xi2, theta, kernel):

    r"""
    Computes K_{ff}\left( \xi, \xi' \right).

    Corresponds to equation 17 from BNSE paper.

    xi1 # [M,]
    xi2 # [M',]
    """

    """
    magnitude = np.pi * sigma**2 / (np.sqrt(alpha*(alpha + 2*gamma)))
    return magnitude * np.exp(-np.pi**2/(2*alpha)*outersum(x,-y)**2 - 2*np.pi*2/(alpha + 2*gamma)*(outersum(x,y)/2-theta)**2)
    """

    #NOTE -- this will only work for 1D data!
    _alpha = tf.reshape(kernel.alpha, [-1,]) # [1,]
    _gamma = tf.reshape(kernel.bandwidths, [-1,]) # [Q,]
    _sigma = tf.reshape(kernel.powers, [-1,]) # [Q,]
    _theta = tf.reshape(theta, [-1,]) # [Q,]

    _pi = np.pi

    magnitude = _pi * _sigma**2 / (tf.sqrt(_alpha * (_alpha + 2.*_gamma))) # [Q,]
    magnitude = magnitude[tf.newaxis, :, tf.newaxis] # [1, Q, 1]

    Kxi_xi = tf.math.exp(-_pi**2/(2.*_alpha[tf.newaxis, :, tf.newaxis]) * #  [1, 1, 1]
                            tf.square(tf.reshape(xi1, [-1,1]) - tf.reshape(xi2,[1,-1]))[:,tf.newaxis,:] # [M, 1, M']
                            ) # [M, 1, M']
    Kxi_xi *= tf.math.exp( - 2.*_pi**2/
                            (_alpha + 2.*_gamma)[tf.newaxis, :, tf.newaxis] # [1, Q, 1]
                            * tf.square((tf.reshape(xi1, [-1,1]) + tf.reshape(xi2,[1,-1]))[:,tf.newaxis,:]/2. # [M, 1, M']
                                        -tf.reshape(_theta,[1,-1,1]) # [1, Q, 1]
                                        ) # [M, Q, M']
                            ) # [M, Q, M']
    
    return tf.reduce_sum(magnitude * Kxi_xi, axis=1) # [M, M']

#TODO -- seems to not work with ``Parameters''
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
