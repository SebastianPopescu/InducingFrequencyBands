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

from ..config import default_float
from ..inducing_variables import InducingPatches, InducingPoints, Multiscale, SpectralInducingVariables
from ..kernels import Convolutional, Kernel, SquaredExponential, MultipleSpectralBlock, SpectralKernel 
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
    #NOTE -- not sure for which model this could be used
    """
    
    Kzz = kernel(inducing_variable.Z)
    Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)
    return Kzz

@Kuu.register(SpectralInducingVariables, MultipleSpectralBlock)
def Kuu_block_multi_spectral_kernel_inducingpoints(
    inducing_variable: SpectralInducingVariables, kernel: MultipleSpectralBlock, *, jitter: float = 0.0
) -> tf.Tensor:
    
    Kzz = tf.linalg.diag(kernel.powers)
    Kzz = tf.cast(Kzz, default_float())

    return Kzz

    
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
