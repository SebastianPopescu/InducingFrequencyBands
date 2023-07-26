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
from ..kernels import Convolutional, Kernel, SquaredExponential, MultipleSpectralBlock, SpectralKernel, Matern12 
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


@Kuu.register(SpectralInducingVariables, Matern12)
def Kuu_L2_features_spectral_kernel_inducingpoints(
    inducing_variable: SpectralInducingVariables, kernel: Matern12, *, jitter: float = 0.0
) -> tf.Tensor:
    
    """
    To be used for the L2 features case of VFF.
    """
    print('---- inside Kuu ------')
    lamb = 1.0 / kernel.lengthscales
    a = inducing_variable.a 
    b = inducing_variable.b 
    omegas = inducing_variable.omegas # shape - [M, ]   
    #omegas = tf.reshape(omegas, [-1,1]) # shape - [M, 1]   
    #spectrum = inducing_variable.spectrum(kernel) # shape - [M, ] 

    #cosine block
    Kzz_cosine = -(2. * lamb**2) # shape - [1,]
    Kzz_cosine *= 1. - tf.math.exp(lamb * (a-b)) # shape - [1,]
    Kzz_cosine /= lamb**2 + tf.reshape(omegas**2,[-1,1]) # shape - [M, 1]
    Kzz_cosine /= tf.reshape(lamb**2 + omegas**2, [1,-1]) # shape - [M,M]
    print('cosine block')
    print(Kzz_cosine)

    #addition of diagonal specific terms to cosine block
    diagonal_cosine = (b-a) * lamb
    diagonal_cosine /= lamb**2 + omegas**2
    #specific addition just for the first cosine of the harmonics
    scaling_list = [2.]
    upto = tf.shape(omegas)[0] - 1
    scaling_list.extend([1. for _ in range(upto)])
    #FIXME -- tmp workaround
    Kzz_cosine += tf.cast(tf.linalg.diag(diagonal_cosine), default_float()) * tf.cast(
        tf.linalg.diag(scaling_list), default_float()) #  shape - [M,M]
    #Kzz_cosine += tf.cast(tf.linalg.diag(diagonal_cosine), default_float())
    print('cosine block')
    print(Kzz_cosine)

    #sine block
    Kzz_sine = 2. * tf.reshape(omegas[omegas != 0], [-1,1]) * tf.reshape(omegas[omegas != 0], [1,-1]) # shape - [M-1, M-1]
    Kzz_sine *= 1. - tf.math.exp(lamb * (a-b))
    Kzz_sine /= lamb**2 + tf.reshape(omegas[omegas != 0], [-1,1])**2
    Kzz_sine /= lamb**2 + tf.reshape(omegas[omegas != 0], [1,-1])**2
    print('sine block')
    print(Kzz_sine)

    #addition of diagonal specific terms to sine block
    diagonal_sine = (b - a) * lamb 
    diagonal_sine /= lamb**2 + omegas[omegas != 0]**2
    Kzz_sine += tf.linalg.diag(diagonal_sine) #  shape - [M-1, M-1]
    print('sine block')
    print(Kzz_sine)
    operator = tf.linalg.LinearOperatorBlockDiag([tf.linalg.LinearOperatorFullMatrix(Kzz_cosine), 
                                                  tf.linalg.LinearOperatorFullMatrix(Kzz_sine)])
    Kzz = operator.to_dense() # shape - [2M-1, 2M-1]
    print('Kzz')
    print(Kzz)

    Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)
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
