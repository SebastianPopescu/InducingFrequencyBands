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

from ..config import default_float
from ..base import TensorLike, TensorType
from ..inducing_variables import (
    InducingPatches, 
    InducingPoints, 
    Multiscale, 
    SpectralInducingVariables, 
    SymRectangularSpectralInducingPoints, 
    AsymRectangularSpectralInducingPoints,
    AsymDiracSpectralInducingPoints,
)
from ..kernels import (
    Convolutional, 
    Kernel, 
    SquaredExponential, 
    MultipleSpectralBlock, 
    SpectralKernel, 
    DecomposedMultipleSpectralBlock,
    DecomposedMultipleDiracSpectralBlock
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


@Kuf.register(SymRectangularSpectralInducingPoints, MultipleSpectralBlock, TensorLike)
#TODO -- re-introduce the check_shapes 
#@check_shapes(
#    "inducing_variable: [M, D, 1]",
#    "Xnew: [batch..., N, D]",
#    "return: [M, batch..., N]",
#)
def Kuf_sym_block_spectral_kernel_inducingpoints(
    inducing_variable: SymRectangularSpectralInducingPoints, kernel: MultipleSpectralBlock, 
    Xnew: TensorType
) -> tf.Tensor:

    """
    Implies having inter-domain inducing points with just symmetrical rectangular 
    function in their spectrum. Results in the elimination of the imaginary component.
    """

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
@Kuf.register(AsymRectangularSpectralInducingPoints, DecomposedMultipleSpectralBlock, TensorLike)
#TODO -- re-introduce the check_shapes 
#@check_shapes(
#    "inducing_variable: [M, D, 1]",
#    "Xnew: [batch..., N, D]",
#    "return: [M, batch..., N]",
#)
def Kuf_asym_block_spectral_kernel_inducingpoints(
    inducing_variable: AsymRectangularSpectralInducingPoints, 
    kernel: DecomposedMultipleSpectralBlock, 
    Xnew: TensorType
) -> tf.Tensor:

    """
    Implies having inter-domain inducing points with just one rectangular 
    function in their spectrum. In practice, we also use the symmetrical rectangular
    function but as a different inter-domain inducing point.
    """

    _means = kernel.means # expected shape [D, M]
    _bandwidths = kernel.bandwidths # expected shape [D, M]
    _real_powers = kernel.real_powers # expected shape [M, ]
    _img_powers = kernel.img_powers # expected shape [M, ]

    #################
    ### real part ###
    #################

    ### positive frequencies 

    r_sine_term = tf.reduce_prod( tf.sin(tf.cast(np.pi, default_float()) * tf.multiply(
        tf.transpose(_bandwidths)[..., None], # [M, D, 1]
        tf.transpose(Xnew)[None, ...] # [1, D, N]
    ) #[M, D, N]
    ), axis = 1) #[M, N]
    
    r_pos_cosine_term = tf.reduce_prod( tf.cos(2. * tf.cast(np.pi, default_float()) * 
                                               tf.multiply(tf.transpose(_means)[..., None], # [M, D, 1]
        tf.transpose(Xnew)[None, ...] # [1, D, N]
    ) #[M, D, N]
    ), axis = 1) #[M, N]

    #NOTE -- this is the case where we pre-multiply the inter-domain
    # inducing points with $\sqrt{\alpha}$ in the definition.
    r_pre_multiplier = _real_powers * tf.reduce_prod(
        tf.math.reciprocal(2. * _bandwidths), axis = 0) # expected shape (M, )
    r_pre_multiplier *= tf.cast(np.pi, default_float()) 
    #NOTE -- uncomment this to retrive the original formulation of inter-domain inducing points
    #r_pre_multiplier /= tf.cast(tf.sqrt(kernel.alpha), default_float())

    pos_real_part  = r_pre_multiplier[..., None] * r_sine_term * r_pos_cosine_term # expected shape (M, N)

    ### negative frequencies

    #NOTE -- the negative sign comes from the fact that the sine is an odd function
    neg_real_part  = -r_pre_multiplier[..., None] * r_sine_term * r_pos_cosine_term # expected shape (M, N)

    real_part = tf.concat([pos_real_part, neg_real_part], axis = 0)


    ######################
    ### imaginary part ###
    ######################

    ### positive frequencies 

    #NOTE -- this is the case where we pre-multiply the inter-domain
    # inducing points with $\sqrt{\alpha}$ in the definition.
    i_pre_multiplier = _img_powers * tf.reduce_prod(
        tf.math.reciprocal(2. * _bandwidths), axis = 0) # expected shape (M, )
    i_pre_multiplier *= tf.cast(np.pi, default_float()) 
    #NOTE -- uncomment this to retrive the original formulation of inter-domain inducing points
    #i_pre_multiplier /= tf.cast(tf.sqrt(kernel.alpha), default_float())

    i_pos_sine_term = tf.reduce_prod( tf.sin( 2. * tf.cast(np.pi, default_float())
                                             * tf.multiply(tf.transpose(_means)[..., None], # [M, D, 1]
        tf.transpose(Xnew)[None, ...] # [1, D, N]
    ) #[M, D, N]
    ), axis = 1) #[M, N]

    pos_img_part  = i_pre_multiplier[..., None] * r_sine_term * i_pos_sine_term # expected shape (M, N)

    # negative frequencies

    #NOTE -- the negative sign comes from the fact that the sine is an odd function
    neg_img_part  = - i_pre_multiplier[..., None] * r_sine_term * i_pos_sine_term # expected shape (M, N)

    img_part = tf.concat([pos_img_part, neg_img_part], axis = 0)

    #NOTE -- this is the case when we are taking just the positive frequencies.
    #Kzf = tf.concat([pos_real_part, pos_img_part], axis = 0)
    Kzf = tf.concat([real_part, img_part], axis = 0)

    return Kzf



#TODO -- need to introduce these
@Kuf.register(AsymDiracSpectralInducingPoints, DecomposedMultipleDiracSpectralBlock, TensorLike)
#TODO -- re-introduce the check_shapes 
#@check_shapes(
#    "inducing_variable: [M, D, 1]",
#    "Xnew: [batch..., N, D]",
#    "return: [M, batch..., N]",
#)
def Kuf_asym_dirac_block_spectral_kernel_inducingpoints(
    inducing_variable: AsymDiracSpectralInducingPoints, 
    kernel: DecomposedMultipleDiracSpectralBlock, 
    Xnew: TensorType
) -> tf.Tensor:

    """
    Implies having inter-domain inducing points with just one rectangular 
    function in their spectrum. In practice, we also use the symmetrical rectangular
    function but as a different inter-domain inducing point.
    """

    _means = kernel.means # expected shape [D, M]
    _bandwidths = kernel.bandwidths # expected shape [D, M]
    _real_powers = kernel.real_powers # expected shape [M, ]
    _img_powers = kernel.img_powers # expected shape [M, ]

    #################
    ### real part ###
    #################

    ### positive frequencies 

    r_pos_cosine_term = tf.cos(2. * tf.cast(np.pi, default_float()) * 
                               tf.multiply(tf.transpose(_means)[..., None], # [M, D, 1]
                                           tf.transpose(Xnew)[None, ...] # [1, D, N]
                                )) #[M, D, N] -- in this case D=1

    r_pos_cosine_term = tf.transpose(r_pos_cosine_term, [1,0,2]) #[D, M, N] -- in this case D=1

    sq_diff = tf.square(tf.reshape(_means, [-1, 1]) - tf.reshape(_means, [1, -1])) # [M, M]

    exp1 = tf.math.exp(tf.cast(-tf.math.reciprocal(4. * kernel.alpha), default_float())
                        * tf.cast(sq_diff, default_float()))
    exp1 = exp1[..., tf.newaxis] # [M, M, 1]

    _real_powers = tf.reshape(_real_powers, [-1, 1, 1]) # [M, 1, 1]   

    pos_real_part = _real_powers * 0.5 * r_pos_cosine_term * exp1 # [M, M, N]
    pos_real_part = tf.reduce_sum(pos_real_part, axis = 0) # [M, N]

    _pre_multiplier = tf.sqrt(tf.cast(np.pi, default_float()))
    _pre_multiplier /= (2. * kernel.alpha)

    pos_real_part *= _pre_multiplier

    ### negative frequencies
    # NOTE -- I think in this case both are equal
    #real_part = tf.concat([pos_real_part, pos_real_part], axis = 0)


    ######################
    ### imaginary part ###
    ######################

    ### positive frequencies 

    r_pos_sine_term = tf.sin(2. * tf.cast(np.pi, default_float()) * 
                               tf.multiply(tf.transpose(_means)[..., None], # [M, D, 1]
                                           tf.transpose(Xnew)[None, ...] # [1, D, N]
                                )) #[M, D, N] -- in this case D=1
    r_pos_sine_term = tf.transpose(r_pos_sine_term, [1,0,2]) #[D, M, N] -- in this case D=1

    _img_powers = tf.reshape(_img_powers, [-1, 1, 1]) # [M, 1, 1]   
    pos_img_part = _img_powers * 0.5 * r_pos_sine_term * exp1 # [M, M, N]
    pos_img_part = tf.reduce_sum(pos_img_part, axis = 0) # [M, N]
    pos_img_part *= _pre_multiplier

    ### negative frequencies
    #NOTE -- I think in this case it's just the negative
    #img_part = tf.concat([pos_img_part, -pos_img_part], axis = 0)

    #NOTE -- this is the case when we are taking just the positive frequencies.
    Kzf = tf.concat([pos_real_part, pos_img_part], axis = 0)
    #Kzf = tf.concat([real_part, img_part], axis = 0)
    print('Kzf')
    print(tf.shape(Kzf))

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
