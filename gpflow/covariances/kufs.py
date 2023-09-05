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

import numpy as np  # pylint: disable=unused-import  # Used by Sphinx to generate documentation.
import tensorflow as tf
from check_shapes import check_shapes
import math

from ..config import default_float
from ..base import TensorLike, TensorType
from ..inducing_variables import (
    InducingPatches, 
    InducingPoints, 
    Multiscale, 
    SymRectangularSpectralInducingPoints,
    IFFRectangularSpectralInducingPoints,
)
     
from ..kernels import Convolutional, Kernel, SquaredExponential, MultipleSpectralBlock, SpectralKernel, MixtureSpectralGaussianVectorized
from .dispatch import Kuf


@Kuf.register(InducingPoints, Kernel, TensorLike)
@check_shapes(
    "inducing_variable: [M, D, 1]",
    "Xnew: [batch..., N, D]",
    "return: [M, batch..., N]",
)
def Kuf_kernel_inducingpoints(
    inducing_variable: InducingPoints, kernel: Kernel, Xnew: TensorType
) -> tf.Tensor:
    return kernel(inducing_variable.Z, Xnew)


@Kuf.register(IFFRectangularSpectralInducingPoints, SquaredExponential, TensorLike)
#@check_shapes(
#    "inducing_variable: [M, D, 1]",
#    "Xnew: [batch..., N, D]",
#    "return: [M, batch..., N]",
#)
def Kuf_IFF_inducingpoints(
    inducing_variable: IFFRectangularSpectralInducingPoints, 
    kernel: SquaredExponential, Xnew: TensorType
) -> tf.Tensor:
    
    prod = 2. * math.pi * tf.reshape(Xnew, [1,-1]) * tf.reshape(inducing_variable.Z, [-1,1])

    out_cosine = tf.cast(tf.sqrt(2.), default_float()) * tf.math.cos(prod)
    out_sine = tf.cast(tf.sqrt(2.), default_float()) * tf.math.sin(prod)

    return tf.concat([out_cosine, out_sine], axis = 0)



@Kuf.register(InducingPoints, MultipleSpectralBlock, TensorLike)
@check_shapes(
    "inducing_variable: [M, D, 1]",
    "Xnew: [batch..., N, D]",
    "return: [M, batch..., N]",
)
def Kuf_spectral_kernel_inducingpoints(
    inducing_variable: InducingPoints, kernel: MultipleSpectralBlock, Xnew: TensorType
) -> tf.Tensor:
    Kzf = kernel(inducing_variable.Z, Xnew)

    return Kzf

@Kuf.register(SymRectangularSpectralInducingPoints, MixtureSpectralGaussianVectorized, TensorLike)
def Kuf_windowed_mixture_gaussian_spectral_kernel_inducingpoints(
    inducing_variable: SymRectangularSpectralInducingPoints, kernel: MixtureSpectralGaussianVectorized, Xnew: TensorType
) -> tf.Tensor:

    pos_freq_real, pos_freq_img = cross_covariance(inducing_variable.Z, Xnew, kernel.means, kernel)
    neg_freq_real, neg_freq_img = cross_covariance(inducing_variable.Z, Xnew, -kernel.means, kernel)

    real_cov = 0.5*(pos_freq_real + neg_freq_real)
    imag_cov = 0.5*(pos_freq_img + neg_freq_img)

    Kuf = tf.concat([real_cov, imag_cov], axis = 0)

    return Kuf

def cross_covariance(xi, t, theta, kernel):

    r"""
    Computes K_{Fy}\left( \xi, t^{*}\right),
    which is the cross-covariance between the time-domain
    and the frequency domain.
    
    Corresponds to Real and Imaginary part of equation ... 
    """
    #NOTE -- this will only work for 1D data!
    _alpha = tf.reshape(kernel.alpha, [-1,]) # [1,]
    _gamma = tf.reshape(kernel.bandwidths, [-1,]) # [Q,]
    _sigma = tf.reshape(kernel.powers, [-1,]) # [Q,]
    _theta = tf.reshape(theta, [-1,]) # [Q,]

    _pi = np.pi

    at = _alpha / _pi**2 # [1,]
    gt = _gamma / _pi**2 # [Q,]
    L = 1./at + 1./gt # [Q,]

    # prepare for broadcasting
    at = at[tf.newaxis, :, tf.newaxis] # [1, 1, 1]
    gt = gt[tf.newaxis, :, tf.newaxis] # [1, Q, 1]
    L = L[tf.newaxis, :, tf.newaxis] # [1, Q, 1]

    Kuf = _sigma[tf.newaxis, :, tf.newaxis]**2 / tf.sqrt(_pi * (at+gt)) # [1,Q,1]

    Kuf*= tf.math.exp(- tf.square(tf.reshape(xi, [-1,1,1])
                                - tf.reshape(_theta, [1,-1,1])
                                ) # [M, Q, 1] 
                        / (at+gt) - tf.square(tf.reshape(t, [1, 1, -1])) # [N, 1, 1]
                    * _pi**2 / L) # [M, Q, 1]

    #NOTE -- why do we have a minus sign inside the cosine?
    Kuf_real = Kuf * tf.math.cos(- 2*_pi* (tf.reshape(xi, [-1,1,1]) / at # [M, Q, 1] 
                        + tf.reshape(_theta, [1,-1,1]) / gt # [1, Q, 1]
                        )
                        / L # [M, Q, 1] 
                        * tf.reshape(t, [1,1,-1]) # [1, 1, N]
                        ) # [M, Q, N]
    Kuf_real = tf.reduce_sum(Kuf_real, axis=1) # [M,N]

    Kuf_img = Kuf * tf.math.sin(- 2*_pi* (tf.reshape(xi, [-1,1,1]) / at # [M, Q, 1] 
                        + tf.reshape(_theta, [1,-1,1]) / gt # [1, Q, 1]
                        )
                        / L # [M, Q, 1] 
                        * tf.reshape(t, [1,1,-1]) # [1, 1, N]
                        ) # [M, Q, N]
    Kuf_img = tf.reduce_sum(Kuf_img, axis=1) # [M,N]

    return Kuf_real, Kuf_img


@Kuf.register(Multiscale, SquaredExponential, TensorLike)
@check_shapes(
    "inducing_variable: [M, D, 1]",
    "Xnew: [batch..., N, D]",
    "return: [M, batch..., N]",
)
def Kuf_sqexp_multiscale(
    inducing_variable: Multiscale, kernel: SquaredExponential, Xnew: TensorType
) -> tf.Tensor:
    Xnew, _ = kernel.slice(Xnew, None)
    Zmu, Zlen = kernel.slice(inducing_variable.Z, inducing_variable.scales)
    idlengthscales = kernel.lengthscales + Zlen
    d = inducing_variable._cust_square_dist(Xnew, Zmu, idlengthscales[None, :, :])
    lengthscales = tf.reduce_prod(kernel.lengthscales / idlengthscales, 1)
    lengthscales = tf.reshape(lengthscales, (1, -1))
    return tf.transpose(kernel.variance * tf.exp(-0.5 * d) * lengthscales)


@Kuf.register(InducingPatches, Convolutional, object)
@check_shapes(
    "inducing_variable: [M, D, 1]",
    "Xnew: [batch..., N, D2]",
    "return: [M, batch..., N]",
)
def Kuf_conv_patch(
    inducing_variable: InducingPatches, kernel: Convolutional, Xnew: TensorType
) -> tf.Tensor:
    Xp = kernel.get_patches(Xnew)  # [N, num_patches, patch_len]
    bigKzx = kernel.base_kernel.K(
        inducing_variable.Z, Xp
    )  # [M, N, P] -- thanks to broadcasting of kernels
    Kzx = tf.reduce_sum(bigKzx * kernel.weights if hasattr(kernel, "weights") else bigKzx, [2])
    return Kzx / kernel.num_patches
