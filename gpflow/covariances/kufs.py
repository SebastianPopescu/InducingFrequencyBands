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
from ..kernels import Convolutional, Kernel, SquaredExponential, MultipleSpectralBlock, SpectralKernel, L2_Matern12, RKHS_Matern12
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

@Kuf.register(SpectralInducingVariables, L2_Matern12, TensorLike)
#FIXME -- re-introduce the check_shapes 
#@check_shapes(
#    "inducing_variable: [M, D, 1]",
#    "Xnew: [batch..., N, D]",
#    "return: [M, batch..., N]",
#)
def Kuf_L2_features_spectral_kernel_inducingpoints(
    inducing_variable: SpectralInducingVariables, kernel: L2_Matern12, Xnew: TensorType
) -> tf.Tensor:

    lamb = 1.0 / kernel.lengthscales
    
    omegas = inducing_variable.omegas # shape - [M, ]   
    spectrum = inducing_variable.spectrum(kernel) # shape - [M, ] 
    a = inducing_variable.a 
    b = inducing_variable.b 
    
    #NOTE -- corresponds to real part of equation 46 from VFF paper.
    real_part = spectrum * tf.math.cos(omegas * (Xnew - a))
    real_part += spectrum * tf.math.reciprocal(2. * lamb) * (lamb * (- tf.math.exp(a - Xnew) 
                                                                     - tf.math.exp(Xnew - b) ))

    #NOTE -- corresponds to imaginary part of equation 46 from VFF paper.
    #NOTE -- we don't want to use zero freq for sine features
    imaginary_part = spectrum[omegas != 0] * tf.math.sin(omegas[omegas != 0] * (Xnew - a))
    imaginary_part += spectrum[omegas != 0]  * tf.math.reciprocal(2. * lamb) * spectrum[omegas != 0] * (tf.math.exp(a - Xnew) 
                                                                              - tf.math.exp(Xnew - b))

    _Kuf = tf.concat([real_part, imaginary_part], 1) 

    #FIXME -- why are we getting the transpose? I don't think it matters that much in the end.
    return tf.transpose(_Kuf) # shape - [2M - 1, N]


@Kuf.register(SpectralInducingVariables, RKHS_Matern12, TensorLike)
#FIXME -- re-introduce the check_shapes 
#@check_shapes(
#    "inducing_variable: [M, D, 1]",
#    "Xnew: [batch..., N, D]",
#    "return: [M, batch..., N]",
#)
def Kuf_RKHS_features_spectral_kernel_inducingpoints(
    inducing_variable: SpectralInducingVariables, kernel: RKHS_Matern12, Xnew: TensorType
) -> tf.Tensor:

    lamb = 1.0 / kernel.lengthscales
    
    omegas = inducing_variable.omegas # shape - [M, ]   
    spectrum = inducing_variable.spectrum(kernel) # shape - [M, ] 
    a = inducing_variable.a 
    b = inducing_variable.b 

    #No edges case.
    Kuf_cos = tf.transpose(tf.cos(omegas * (Xnew- a)))
    omegas = omegas[omegas != 0]  # don't compute zeros freq.
    Kuf_sin = tf.transpose(tf.sin(omegas * (Xnew - a)))
    
    return tf.concat([Kuf_cos, Kuf_sin], axis=0)



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
