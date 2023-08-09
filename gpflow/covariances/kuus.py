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
from ..kernels import Convolutional, Kernel, SquaredExponential, MultipleSpectralBlock, SpectralKernel, L2_Matern12, RKHS_Matern12 
from .dispatch import Kuu
from gpflow.matrix_structures import Rank1Mat, BlockDiagMat

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


@Kuu.register(SpectralInducingVariables, L2_Matern12)
def Kuu_L2_features_spectral_kernel_inducingpoints(
    inducing_variable: SpectralInducingVariables, kernel: L2_Matern12, *, jitter: float = 0.0
) -> tf.Tensor:
    
    """
    To be used for the L2 features case of VFF.
    """

    lamb = 1.0 / kernel.lengthscales
    a = inducing_variable.a 
    b = inducing_variable.b 
    omegas = inducing_variable.omegas # shape - [M, ]   
    #omegas = tf.reshape(omegas, [-1,1]) # shape - [M, 1]   
    #spectrum = inducing_variable.spectrum(kernel) # shape - [M, ] 

    #cosine block -- corresponds to equation 99 in VFF paper.
    Kzz_cosine = -(2. * tf.square(lamb)) # shape - [1,]
    Kzz_cosine *= 1. - tf.math.exp(lamb * (a-b)) # shape - [1,]
    Kzz_cosine /= tf.reshape(tf.square(lamb) + tf.square(omegas), [-1,1]) # shape - [M, 1]
    Kzz_cosine /= tf.reshape(tf.square(lamb) + tf.square(omegas), [1,-1]) # shape - [M, M]

    #addition of diagonal specific terms to cosine block
    # corresponds to equation 100 in VFF paper.
    diagonal_cosine = (b-a) * lamb
    diagonal_cosine /= tf.square(lamb) + tf.square(omegas)
    #specific addition just for the first cosine of the harmonics
    # corresponds to equation 101 in VFF paper.
    scaling_list = [2.]
    upto = tf.shape(omegas)[0] - 1
    #TODO -- this causes an error when trying to train this model. 
    # Probably due to some tf.function underlying mechanics
    scaling_list.extend([1. for _ in range(upto)])
    #FIXME -- tmp workaround
    Kzz_cosine += tf.cast(tf.linalg.diag(diagonal_cosine), default_float()) * tf.cast(
        tf.linalg.diag(scaling_list), default_float()) #  shape - [M,M]
    #Kzz_cosine += tf.cast(tf.linalg.diag(diagonal_cosine), default_float())

    #sine block - corresponds to equation 105 in VFF paper.
    #NOTE -- we don't want to use zero freq for sine features
    Kzz_sine = 2. * tf.reshape(omegas[omegas != 0], [-1,1]) * tf.reshape(
        omegas[omegas != 0], [1,-1]) # shape - [M-1, M-1]
    Kzz_sine *= 1. - tf.math.exp(lamb * (a-b))
    Kzz_sine /= tf.square(lamb) + tf.square(tf.reshape(omegas[omegas != 0], [-1,1]))
    Kzz_sine /= tf.square(lamb) + tf.square(tf.reshape(omegas[omegas != 0], [1,-1]))

    #addition of diagonal specific terms to sine block - corresponds to equation 106 in VFF paper.
    diagonal_sine = (b - a) * lamb 
    diagonal_sine /= tf.square(lamb) + tf.square(omegas[omegas != 0])
    Kzz_sine += tf.linalg.diag(diagonal_sine) #  shape - [M-1, M-1]
    print('sine block')
    print(Kzz_sine)

    Kzz = BlockDiagMat(Kzz_cosine, Kzz_sine)    

    #NOTE -- doesn't seem to help that much
    #Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)
    
    return Kzz


@Kuu.register(SpectralInducingVariables, RKHS_Matern12)
def Kuu_RKHS_features_spectral_kernel_inducingpoints(
    inducing_variable: SpectralInducingVariables, kernel: RKHS_Matern12, *, jitter: float = 0.0
) -> tf.Tensor:
    
    """
    To be used for the RKHS features case of VFF.
    """

    lamb = 1.0 / kernel.lengthscales
    a = inducing_variable.a 
    b = inducing_variable.b 
    omegas = inducing_variable.omegas # shape - [M, ]    
    #spectrum = inducing_variable.spectrum(kernel) # shape - [M, ] 

    # cos part first
    #NOTE -- wtf is this?
    two_or_four = np.where(omegas == 0, 2.0, 4.0)
    
    d_cos = (b - a) * (tf.square(lamb) + tf.square(omegas)) / lamb / kernel.variance / two_or_four
    v_cos = tf.ones(tf.shape(d_cos), default_float()) / tf.sqrt(kernel.variance)

    # now the sin part
    omegas = omegas[omegas != 0]  # don't compute omega=0
    d_sin = (b - a) * (tf.square(lamb) + tf.square(omegas)) / lamb / kernel.variance / 4.0

    return BlockDiagMat(Rank1Mat(d_cos, v_cos), tf.linalg.diag(d_sin))


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
