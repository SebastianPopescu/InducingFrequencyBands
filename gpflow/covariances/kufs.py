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
from ..kernels import Convolutional, Kernel, SquaredExponential,  IFFMultipleSpectralBlock
from .dispatch import Kuf
import math

def rbf_spectral_density(freq, lengthscale, variance):

    _pi = math.pi
    _pi = tf.cast(_pi, tf.float64)

    constant_num = tf.sqrt(tf.sqrt(lengthscale)) / (2. * tf.sqrt(_pi))
    freq_term = tf.exp(- tf.sqrt(lengthscale) * freq**2 * 0.25)
    S = constant_num * freq_term

    return S * variance


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


# NOTE -- this completly breaks the dispatcher method in GPflow
# not sure this can be fixed as the underlying kernel is a squared exponential that 
# needs to be used for Kff
@Kuf.register(SpectralInducingVariables, IFFMultipleSpectralBlock, TensorLike)
#@check_shapes(
#    "inducing_variable: [M, D, 1]",
#    "Xnew: [batch..., N, D]",
#    "return: [M, batch..., N]",
#)
def Kuf_IFF_block_spectral_kernel_inducingpoints(
    inducing_variable: SpectralInducingVariables, kernel: IFFMultipleSpectralBlock, 
    Xnew: TensorType
) -> tf.Tensor:

    _means = kernel.means # expected shape [D, M]
    _lengthscales = kernel.lengthscales # expected shape TODO -- add it
    _variance = kernel.variance # expected shape TODO -- add it
    print('--- inside spectral Kuf dfispatcher ------')
 
    cosine_term = 2.0 * tf.reduce_prod( tf.cos( 2.0 * math.pi * 
                                               tf.multiply(
            tf.transpose(_means)[..., None], # [M, D, 1]
            tf.transpose(Xnew)[None, ...] # [1, D, N]
                                                ) #[M, D, N]
                                                ), axis = 1) #[M, N]
    print('cosine term') 
    print(cosine_term)
    
    #TODO -- this squeeze is a hack, fix underlying issue
    pre_multiplier = tf.squeeze(tf.sqrt( rbf_spectral_density(freq = _means, 
        lengthscale = _lengthscales, variance = _variance)), axis = 0) # expected shape (M, )
    print('pre_multiplier')
    print(pre_multiplier[..., None])

    print('output')
    print(pre_multiplier[..., None] *  cosine_term)

    return pre_multiplier[..., None] * cosine_term # expected shape (M, N)



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
