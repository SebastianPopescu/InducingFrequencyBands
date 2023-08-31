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

@Kuf.register(SymRectangularDiracDeltaSpectralInducingPoints, MultipleDiracDeltaSpectralBlock, TensorLike)
#@check_shapes(
#    "inducing_variable: [M, D, 1]",
#    "Xnew: [batch..., N, D]",
#    "return: [M, batch..., N]",
#)
def Kuf_rectangular_dirac_delta_spectral_kernel_inducingpoints(
    inducing_variable: SymRectangularDiracDeltaSpectralInducingPoints, kernel: MultipleDiracDeltaSpectralBlock, Xnew: TensorType
) -> tf.Tensor:

    #Xnew -- [N,D]
    #kernel.means -- [D,M]

    """
    neg_freq_real_pos_freq, neg_freq_imag_pos_freq = dirac_spectrum_time_cross_covariance(
        -kernel.means, Xnew, kernel, kernel.means)
    neg_freq_real_neg_freq, neg_freq_imag_neg_freq = dirac_spectrum_time_cross_covariance(
        -kernel.means, Xnew, kernel, -kernel.means)
    """
        
    pos_freq_real_pos_freq, pos_freq_imag_pos_freq = dirac_spectrum_time_cross_covariance(
        kernel.means, Xnew, kernel, kernel.means)
    pos_freq_real_neg_freq, pos_freq_imag_neg_freq = dirac_spectrum_time_cross_covariance(
        kernel.means, Xnew, kernel, -kernel.means)

    #neg_freq_real = neg_freq_real_pos_freq + neg_freq_real_neg_freq
    pos_freq_real = pos_freq_real_pos_freq + pos_freq_real_neg_freq

    #neg_freq_imag = neg_freq_imag_pos_freq + neg_freq_imag_neg_freq
    pos_freq_imag = pos_freq_imag_pos_freq + pos_freq_imag_neg_freq

    """
    Kzf = tf.concat([neg_freq_real, pos_freq_real, 
                     neg_freq_imag, pos_freq_imag], 
                     axis = 0)
    """
    Kzf = tf.concat([pos_freq_real, 
                     pos_freq_imag], 
                     axis = 0)


    return Kzf


def dirac_spectrum_time_cross_covariance(xi1, Xnew, kernel, means):

    print('--- inside dirac_spectrum_time_cross_covariance ---')

    pre_multiplier = tf.sqrt(math.pi * tf.cast(tf.math.reciprocal(kernel.alpha), default_float()))

    spectrum_powers = kernel.powers * tf.cast(tf.math.reciprocal(2.), default_float()) 
    spectrum_powers = tf.reshape(spectrum_powers, [-1,1,1]) # [Q, 1, 1]
    print('spectrum_powers')
    print(spectrum_powers)

    exp_freq = tf.math.exp(-math.pi**2 * tf.cast(tf.math.reciprocal(kernel.alpha), default_float())  * 
                           tf.square(tf.reshape(xi1, [1,-1,1]) # [1, M, 1]
                                  - tf.reshape(means, [-1,1,1])) # [Q, 1, 1]
                       ) # [Q, M, 1]
    print('exp_freq')
    print(exp_freq)

    exp_part = spectrum_powers * exp_freq  # [Q, M, 1]

    # real part
    #NOTE -- this will only work with 1D data
    Kzf_real = tf.math.cos(2.* math.pi * tf.transpose(xi1) * tf.transpose(Xnew)) # [M, N]
    Kzf_real = Kzf_real[tf.newaxis, ...] # [1, M, N]
    print('Kzf_real')
    print(Kzf_real)
    Kzf_real *= pre_multiplier * exp_part # [Q, M, N]
    Kzf_real = tf.reduce_sum(Kzf_real, axis = 0) # [M, N]
    print('Kzf_real')
    print(Kzf_real)
    # imaginary part
    Kzf_imag = tf.math.sin(2.* math.pi * tf.transpose(xi1) * tf.transpose(Xnew)) # [M, N]
    Kzf_imag = Kzf_imag[tf.newaxis,...] # [1, M, N]
    Kzf_imag *= pre_multiplier * exp_part # [Q, M, N]
    Kzf_imag = tf.reduce_sum(Kzf_imag, axis = 0) # [M, N]

    return Kzf_real, Kzf_imag



@Kuf.register(SymRectangularSpectralInducingPoints, MultipleSpectralBlock, TensorLike)
#@check_shapes(
#    "inducing_variable: [M, D, 1]",
#    "Xnew: [batch..., N, D]",
#    "return: [M, batch..., N]",
#)
def Kuf_rectangular_spectral_kernel_inducingpoints(
    inducing_variable: SymRectangularSpectralInducingPoints, 
    kernel: MultipleSpectralBlock, 
    Xnew: TensorType
) -> tf.Tensor:

    #Xnew -- [N,D]
    #kernel.means -- [D,M]

    """
    neg_freq_real_pos_freq, neg_freq_imag_pos_freq = dirac_spectrum_time_cross_covariance(
        -kernel.means, Xnew, kernel, kernel.means)
    neg_freq_real_neg_freq, neg_freq_imag_neg_freq = dirac_spectrum_time_cross_covariance(
        -kernel.means, Xnew, kernel, -kernel.means)
    """
        
    pos_freq_real_pos_freq, pos_freq_imag_pos_freq = spectrum_time_cross_covariance(
        kernel.means, Xnew, kernel, kernel.means)
    #pos_freq_real_neg_freq, pos_freq_imag_neg_freq = spectrum_time_cross_covariance(
    #    kernel.means, Xnew, kernel, -kernel.means)

    #neg_freq_real = neg_freq_real_pos_freq + neg_freq_real_neg_freq
    pos_freq_real = pos_freq_real_pos_freq #+ pos_freq_real_neg_freq

    #neg_freq_imag = neg_freq_imag_pos_freq + neg_freq_imag_neg_freq
    pos_freq_imag = pos_freq_imag_pos_freq #+ pos_freq_imag_neg_freq

    """
    Kzf = tf.concat([neg_freq_real, pos_freq_real, 
                     neg_freq_imag, pos_freq_imag], 
                     axis = 0)
    """
    Kzf = tf.concat([pos_freq_real, 
                     pos_freq_imag], 
                     axis = 0)


    return Kzf

def spectrum_time_cross_covariance(xi1, Xnew, kernel, means):

    _bandwidths = kernel.bandwidths # expected shape [D, M]
    _powers = kernel.powers # expected shape [M, ]

    spectrum_powers = _powers * tf.cast(tf.math.reciprocal(2. * _bandwidths), default_float()) 
    spectrum_powers = tf.reshape(spectrum_powers, [-1, 1]) #[M, 1]

    # real part 
    real_sine_term = tf.reduce_prod( tf.sin(math.pi * 
                                            tf.multiply(tf.transpose(_bandwidths)[..., None], # [M, D, 1]
                                                        tf.transpose(Xnew)[None, ...] # [1, D, N]
                                                        ) #[M, D, N]
                                            ), axis = 1) #[M, N]
    
    real_cosine_term = tf.reduce_prod( tf.cos(2. * math.pi * 
                                              tf.multiply(tf.transpose(xi1)[..., None], # [M, D, 1]
                                                          tf.transpose(Xnew)[None, ...] # [1, D, N]
                                                          ) #[M, D, N]
                                            ), axis = 1) #[M, N]

    real_part  = spectrum_powers * real_sine_term * real_cosine_term #[M, N]

    # imaginary part 
    imag_sine_term = 2. * tf.reduce_prod( tf.sin(2. * math.pi * 
                                              tf.multiply(tf.transpose(xi1)[..., None], # [M, D, 1]
                                                          tf.transpose(Xnew)[None, ...] # [1, D, N]
                                                          ) #[M, D, N]
                                            ), axis = 1) #[M, N]

    img_part  = spectrum_powers * real_sine_term * imag_sine_term # expected shape (M, N)

    #NOTE -- this is the case when we are taking just the positive frequencies.
    #Kzf = tf.concat([pos_real_part, pos_img_part], axis = 0)


    return real_part, img_part



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
