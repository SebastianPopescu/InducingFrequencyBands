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

from ..base import TensorLike, TensorType
from ..inducing_variables import InducingPatches, InducingPoints, Multiscale, SpectralInducingVariables
from ..kernels import Convolutional, Kernel, SquaredExponential, MultipleSpectralBlock, SpectralKernel
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

#TODO -- need to introduce these
@Kuf.register(SpectralInducingVariables, MultipleSpectralBlock, TensorLike)
#TODO -- re-introduce the check_shapes 
#@check_shapes(
#    "inducing_variable: [M, D, 1]",
#    "Xnew: [batch..., N, D]",
#    "return: [M, batch..., N]",
#)
def Kuf_sym_block_spectral_kernel_inducingpoints(
    inducing_variable: SpectralInducingVariables, kernel: MultipleSpectralBlock, Xnew: TensorType
) -> tf.Tensor:

    _means = kernel.means # expected shape [D, M]
    _bandwidths = kernel.bandwidths # expected shape [D, M]
    _powers = kernel.powers # expected shape [M, ]

    sine_term = tf.reduce_prod( tf.sin(0.5 * tf.multiply(tf.transpose(_bandwidths)[..., None], # [M, D, 1]
        tf.transpose(Xnew)[None, ...] # [1, D, N]
    ) #[M, D, N]
    ), axis = 1) #[M, N]
    
    cosine_term = tf.reduce_prod( tf.cos( tf.multiply(tf.transpose(_means)[..., None], # [M, D, 1]
        tf.transpose(Xnew)[None, ...] # [1, D, N]
    ) #[M, D, N]
    ), axis = 1) #[M, N]

    pre_multiplier = 2. * _powers * tf.reduce_prod(tf.math.reciprocal(_bandwidths), axis = 0) # expected shape (M, )

    Kzf  = pre_multiplier[..., None] * sine_term * cosine_term # expected shape (M, N)

    return Kzf

#TODO -- need to introduce these
@Kuf.register(SpectralInducingVariables, MultipleSpectralBlock, TensorLike)
#TODO -- re-introduce the check_shapes 
#@check_shapes(
#    "inducing_variable: [M, D, 1]",
#    "Xnew: [batch..., N, D]",
#    "return: [M, batch..., N]",
#)
def Kuf_asym_block_spectral_kernel_inducingpoints(
    inducing_variable: SpectralInducingVariables, kernel: MultipleSpectralBlock, Xnew: TensorType
) -> tf.Tensor:

    _means = kernel.means # expected shape [D, M]
    _bandwidths = kernel.bandwidths # expected shape [D, M]
    _powers = kernel.powers # expected shape [M, ]

    #real part

    r_sine_term = tf.reduce_prod( tf.sin(0.5 * tf.multiply(tf.transpose(_bandwidths)[..., None], # [M, D, 1]
        tf.transpose(Xnew)[None, ...] # [1, D, N]
    ) #[M, D, N]
    ), axis = 1) #[M, N]
    
    r_cosine_term = tf.reduce_prod( tf.cos( tf.multiply(tf.transpose(_means)[..., None], # [M, D, 1]
        tf.transpose(Xnew)[None, ...] # [1, D, N]
    ) #[M, D, N]
    ), axis = 1) #[M, N]

    r_pre_multiplier = _powers * tf.reduce_prod(tf.math.reciprocal(_bandwidths), axis = 0) # expected shape (M, )

    real_part  = r_pre_multiplier[..., None] * r_sine_term * r_cosine_term # expected shape (M, N)

    i_sine_term = tf.reduce_prod( tf.sin( tf.multiply(tf.transpose(_means)[..., None], # [M, D, 1]
        tf.transpose(Xnew)[None, ...] # [1, D, N]
    ) #[M, D, N]
    ), axis = 1) #[M, N]
    i_pre_multiplier = - _powers * tf.reduce_prod(tf.math.reciprocal(_bandwidths), axis = 0) # expected shape (M, )

    img_part  = i_pre_multiplier[..., None] * r_sine_term * i_sine_term # expected shape (M, N)

    Kzf = tf.concat([real_part, img_part], axis = 0)

    return Kzf


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
