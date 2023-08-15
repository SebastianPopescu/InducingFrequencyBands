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
    DecomposedMultipleDiracSpectralBlock,
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

#####################################################
##Helper functions for approximating the real Kuu ###
def midpoint_rule(x, std, a, b, n, Burmann_series):

    # Calculate the width of each subinterval
    h = (b - a) / n
    
    # Calculate the midpoint values for each subinterval
    mean_midpoints = tf.linspace(a + h/2, b - h/2, n)

    # Evaluate the function at the midpoint values

    if Burmann_series:
        #NOTE -- this is usign the Burmann series second order approximation
        f_midpoints = burmann_series_approx_erf(x, mean_midpoints, std)
    else:
        #NOTE -- this is using the default Tensorflow version
        f_midpoints = 0.5*(1.0+tf.math.erf((x[tf.newaxis,...]-mean_midpoints)/(tf.cast(tf.math.sqrt(2.),
                                                               default_float())*std)))

    print('f_midpoints')
    print(f_midpoints)
    # Calculate the approximate integral using the midpoint rule formula
    integral_approx = h * tf.reduce_sum(f_midpoints, axis = 0)
    
    return integral_approx


def burmann_series_approx_erf(x, mean, std):

    #\operatorname{erf}\left( \frac{w_{1}-w'}{\sqrt{\frac{2\alpha}{\pi^{2}}}} \right) 
    res = burmann_series_second_order_approx((x-mean)/(tf.cast(tf.math.sqrt(2.),
                                                               default_float())*std))

    return res


def burmann_series_second_order_approx(x):

    """
    Implements a Burmann Series approximation of order 2 to the ERF function
    
    Latex math code:

    \operatorname{erf}\left( x \right) \approx \frac{2}{\sqrt{\pi}}
    \operatorname{sgn}\left( x \right) \sqrt{1 - \exp{-x^{2}}} 
    \left[ \frac{\sqrt{\pi}}{2} + \frac{31}{200}\exp{-x^{2}} 
    - \frac{341}{8000}\exp{-2x^{2}}\right]   
    """

    res = tf.cast(2./tf.sqrt(np.pi), default_float()) * tf.math.sign(x) * tf.cast(tf.sqrt(1. - tf.math.exp(-x**2)), default_float())
    res *= (np.pi/2. + 31./200. * tf.math.exp(-x**2) - 341./8000. * tf.math.exp(-2.*x**2))
    return res


# Helper function -- implements the approximation for a single spectral block for Kuu.
#NOTE -- this uses the Fourier transform of \exp^{\alpha x^{2}} so these results
# are only okay for the symmetrical blocks case.
def Kuu_single_block_symmetrical_multi_spectral_kernel_inducingpoints(
    inducing_variable, kernel, means, powers, *, jitter: float = 0.0):
    
    #NOTE -- this is the unwindowed case
    #Kzz = tf.linalg.diag(kernel.powers)
    #Kzz = tf.cast(Kzz, default_float())

    #NOTE -- implements the following integral but approximated numerically
    #NOTE -- erf is implemented via the second order Burmann series approximation
    #TODO -- do I need  to use tf.stop_gradients here?

    num_approx_N = 1000

    Kzz = tf.linalg.diag(powers / (2. * kernel.bandwidths))

    lower_limit = means - 0.5 * kernel.bandwidths
    upper_limit = means + 0.5 * kernel.bandwidths
    
    num_int_approx = midpoint_rule(upper_limit, std = tf.cast(tf.math.sqrt(kernel.alpha)
                                                              , default_float()) / 
                                   tf.cast(np.pi, default_float()), 
                                   a = lower_limit, b = upper_limit, n = num_approx_N,
                                   Burmann_series=False,
                                   )
    
    num_int_approx -= midpoint_rule(lower_limit, std = tf.cast(tf.math.sqrt(kernel.alpha)
                                                               , default_float()) / 
                                    tf.cast(np.pi,default_float()), 
                                   a = lower_limit, b = upper_limit, n = num_approx_N,
                                   Burmann_series=False,
                                   )
    
    Kzz *= tf.linalg.diag(num_int_approx)

    return tf.squeeze(Kzz, axis = 0)


#NOTE -- this uses the Fourier cosine/sine transform of \exp^{\alpha x^{2}} so these 
# results are only okay for the asymmetrical blocks case.
def Kuu_single_block_asymmetrical_multi_spectral_kernel_inducingpoints_diagonal_version(
    inducing_variable, kernel, means, powers, *, jitter: float = 0.0):
    
    #NOTE -- this is the unwindowed case
    #Kzz = tf.linalg.diag(kernel.powers)
    #Kzz = tf.cast(Kzz, default_float())

    #NOTE -- implements the following integral but approximated numerically
    #NOTE -- erf is implemented via the second order Burmann series approximation
    #TODO -- do I need  to use tf.stop_gradients here?

    num_approx_N = 1000
    
    #NOTE -- original formulation
    #Kzz = tf.linalg.diag( tf.square(tf.cast(np.pi, default_float())) * powers 
    #                     / (2. * kernel.bandwidths * kernel.alpha))

    #NOTE -- this is the case where we pre-multiply the inter-domain
    # inducing points with $\sqrt{\alpha}$ in the definition.
    Kzz = tf.linalg.diag( tf.square(tf.cast(np.pi, default_float())) * powers 
                         / (2. * kernel.bandwidths))

    lower_limit = means - 0.5 * kernel.bandwidths
    upper_limit = means + 0.5 * kernel.bandwidths
    
    num_int_approx = midpoint_rule(upper_limit, std = tf.cast(tf.math.sqrt(2.*kernel.alpha)
                                                              , default_float()) / 
                                   tf.cast(np.pi, default_float()), 
                                   a = lower_limit, b = upper_limit, n = num_approx_N,
                                   Burmann_series=False,
                                   )
    
    num_int_approx -= midpoint_rule(lower_limit, std = tf.cast(tf.math.sqrt(2.*kernel.alpha)
                                                               , default_float()) / 
                                    tf.cast(np.pi,default_float()), 
                                   a = lower_limit, b = upper_limit, n = num_approx_N,
                                   Burmann_series=False,
                                   )
    
    Kzz *= tf.linalg.diag(num_int_approx)

    return tf.squeeze(Kzz, axis = 0)


#helper function
def update_off_diagonal(tensor, bandwidth_powers):
    shape = tf.shape(tensor)
    rows, cols = shape[0], shape[1]
    
    #get indces of elements
    indices = tf.where(tf.ones_like(tensor, dtype=tf.bool))

    #get the right spectral bandwidth for the combination of i,j in Kuu
    desired_band = tf.cast(tf.reduce_sum(indices, axis = -1), tf.float32) * 0.5
    desired_band = tf.cast(desired_band, tf.int32)

    #TODO -- can this be vectorized?
    updates = [bandwidth_powers[_] for _ in desired_band]
    
    # Update elements in even and odd positions of Kuu in the off-diagonal
    # FIXME -- this should update just the off-diagonal elements only    
    updated_tensor = tf.tensor_scatter_nd_update(tensor, indices, updates)

    return updated_tensor

#NOTE -- this uses the Fourier cosine/sine transform of \exp^{\alpha x^{2}} so these 
# results are only okay for the asymmetrical blocks case.
def Kuu_single_block_asymmetrical_multi_spectral_kernel_inducingpoints_full_version(
    inducing_variable, kernel, means, powers, *, jitter: float = 0.0):
    
    """
    Attempt at implementing a full Kuu matrix, with non-zero off-diagonals.
    """

    #NOTE -- this is the unwindowed case
    #Kzz = tf.linalg.diag(kernel.powers)
    #Kzz = tf.cast(Kzz, default_float())

    #NOTE -- implements the following integral but approximated numerically
    #NOTE -- erf is implemented via the second order Burmann series approximation
    #TODO -- do I need  to use tf.stop_gradients here?

    num_approx_N = 20

    #NOTE -- this is the case where we pre-multiply the inter-domain
    # inducing points with $\sqrt{\alpha}$ in the definition.
    Kzz = tf.linalg.diag( tf.square(tf.cast(np.pi, default_float())) * powers 
                         / (2. * kernel.bandwidths * kernel.alpha))
    print('-- Kzz just diagonal ---')
    print(Kzz)
    Kzz = update_off_diagonal(Kzz, powers) * tf.square(tf.cast(np.pi, default_float())) / (2. * kernel.bandwidths * kernel.alpha)
    print('--- Kzz after off-diagonal update ---')
    print(Kzz)

    lower_limit = means - 0.5 * kernel.bandwidths
    upper_limit = means + 0.5 * kernel.bandwidths
    
    num_int_approx = midpoint_rule(tf.reshape(upper_limit,[-1,1]), std = tf.cast(tf.math.sqrt(2.*kernel.alpha)
                                                              , default_float()) / 
                                    tf.cast(np.pi, default_float()), 
                                    a = tf.reshape(lower_limit, [1,-1]), b = tf.reshape(upper_limit, [1,-1]), 
                                    n = num_approx_N,
                                    Burmann_series=False,
                                    )
    print('---- first num_int_approx - -----')
    print(num_int_approx)
    
    second_num_int_approx = midpoint_rule(tf.reshape(lower_limit,[-1,1]), std = tf.cast(tf.math.sqrt(2.*kernel.alpha)
                                                               , default_float()) / 
                                    tf.cast(np.pi,default_float()), 
                                    a = tf.reshape(lower_limit, [1,-1]), b = tf.reshape(upper_limit, [1,-1]),
                                    n = num_approx_N,
                                    Burmann_series=False,
                                    )
    print('---- second num_int_approx - -----')
    print(second_num_int_approx)
    num_approx_N -= second_num_int_approx
    #Kzz *= num_int_approx

    print('---- Kzz  -----')
    print(Kzz)

    return tf.squeeze(Kzz, axis = 0)


def BlockDiagMat(A, B):

    tl_shape = tf.stack([A.shape[0], B.shape[1]])
    br_shape = tf.stack([B.shape[0], A.shape[1]])
    top = tf.concat([A, tf.zeros(tl_shape, default_float())], axis=1)
    bottom = tf.concat([tf.zeros(br_shape, default_float()), B], axis=1)

    return tf.concat([top, bottom], axis=0)


@Kuu.register(SymRectangularSpectralInducingPoints, MultipleSpectralBlock)
def Kuu_sym_block_multi_spectral_kernel_inducingpoints(
    inducing_variable: SymRectangularSpectralInducingPoints, kernel: MultipleSpectralBlock, *, jitter: float = 0.0
) -> tf.Tensor:
    
    #NOTE -- this is the unwindowed case
    #Kzz = tf.linalg.diag(kernel.powers)
    #Kzz = tf.cast(Kzz, default_float())

    Kzz = Kuu_single_block_symmetrical_multi_spectral_kernel_inducingpoints(
        inducing_variable, kernel, kernel.means, kernel.powers)
    #TODO -- fix this little hack, the midpoint rule is giving me a leading 1 dimension
    #NOTE -- doubling the results should account for the two symmetrical rectangular functions.
    #TODO -- need to double check the maths here to be sure that a doubling is sufficient.
    return 2. * Kzz

@Kuu.register(AsymRectangularSpectralInducingPoints, DecomposedMultipleSpectralBlock)
def Kuu_asym_block_multi_spectral_kernel_inducingpoints(
    inducing_variable: AsymRectangularSpectralInducingPoints, kernel: DecomposedMultipleSpectralBlock, *, jitter: float = 0.0
) -> tf.Tensor:

    # Real features corresponding to the cosine transform
    r_Kzz_positive_freq = Kuu_single_block_asymmetrical_multi_spectral_kernel_inducingpoints_diagonal_version(
        inducing_variable, kernel, kernel.means, kernel.real_powers)
    r_Kzz_negative_freq = Kuu_single_block_asymmetrical_multi_spectral_kernel_inducingpoints_diagonal_version(
        inducing_variable, kernel, -kernel.means, kernel.real_powers)

    #NOTE -- I have to be careful in Kuf to maintain the positive, negative freq ordering
    r_Kzz = BlockDiagMat(r_Kzz_positive_freq, r_Kzz_negative_freq)
    
    # Real features corresponding to the sine transform
    i_Kzz_positive_freq = Kuu_single_block_asymmetrical_multi_spectral_kernel_inducingpoints_diagonal_version(
        inducing_variable, kernel, kernel.means, kernel.img_powers)
    i_Kzz_negative_freq = Kuu_single_block_asymmetrical_multi_spectral_kernel_inducingpoints_diagonal_version(
        inducing_variable, kernel, -kernel.means, kernel.img_powers)

    #NOTE -- I have to be careful in Kuf to maintain the positive, negative freq ordering
    i_Kzz = BlockDiagMat(i_Kzz_positive_freq, i_Kzz_negative_freq)

    #NOTE -- I have to be careful in Kuf to maintain the real, imaginary features ordering

    #NOTE -- this is the unwindowed case
    #r_Kzz = BlockDiagMat(tf.linalg.diag(kernel.real_powers), tf.linalg.diag(kernel.real_powers))
    #i_Kzz = BlockDiagMat(tf.linalg.diag(kernel.img_powers), tf.linalg.diag(kernel.img_powers))
    
    #NOTE -- this is the version just with positive frequencies
    #return BlockDiagMat(r_Kzz_positive_freq, i_Kzz_positive_freq)
    
    Kzz = BlockDiagMat(r_Kzz, i_Kzz)
    #Kzz = BlockDiagMat(r_Kzz_positive_freq, i_Kzz_positive_freq)

    return Kzz

def Kuu_single_block_asymmetrical_multi_dirac_spectral_kernel_inducingpoints(
    inducing_variable, kernel, means, powers, *, jitter: float = 0.0):
    
    """
    Attempt at implementing a full Kuu matrix, with non-zero off-diagonals
    in the case of Dirac delta spectral bands.
    """

    #FIXME -- solve this tf.cast
    exp1 = tf.math.exp(-tf.cast(tf.math.reciprocal(4.0 * kernel.alpha), default_float()) * 
                       tf.square(tf.reshape(means, [-1, 1]) - 
                                 tf.reshape(means, [1, -1]) 
                                 )) # [M, M]
    
    sq_diff = tf.square((tf.reshape(means, [-1, 1])[tf.newaxis, ...] + 
                         tf.reshape(means, [1, -1])[tf.newaxis, ...] )* 0.5 - 
                         tf.reshape(means, [-1, 1, 1])) # [M, M, M]

    _powers = tf.reshape(powers, [-1, 1, 1]) # [M, 1, 1]                                 

    #FIXME -- solve this extra tf.cast 
    exp2 =  tf.reduce_sum(_powers * 0.5 * tf.math.exp(-tf.cast(tf.math.reciprocal(4. * kernel.alpha), 
                                                         default_float()) * sq_diff
                                                         ), axis = 0) # [M, M]

    _pre_multiplier = tf.cast(np.pi, default_float())
    _pre_multiplier /= (4. * kernel.alpha**2)

    Kzz = _pre_multiplier * exp1 * exp2

    return Kzz


@Kuu.register(AsymDiracSpectralInducingPoints, DecomposedMultipleDiracSpectralBlock)
def Kuu_asym_dirac_block_multi_spectral_kernel_inducingpoints(
    inducing_variable: AsymDiracSpectralInducingPoints, kernel: DecomposedMultipleDiracSpectralBlock, *, jitter: float = 0.0
) -> tf.Tensor:

    # Real features corresponding to the cosine transform
    r_Kzz_positive_freq = Kuu_single_block_asymmetrical_multi_dirac_spectral_kernel_inducingpoints(
        inducing_variable, kernel, kernel.means, kernel.real_powers)
    r_Kzz_negative_freq = Kuu_single_block_asymmetrical_multi_dirac_spectral_kernel_inducingpoints(
        inducing_variable, kernel, -kernel.means, kernel.real_powers)

    #NOTE -- I have to be careful in Kuf to maintain the positive, negative freq ordering
    r_Kzz = BlockDiagMat(r_Kzz_positive_freq, r_Kzz_negative_freq)
    
    # Real features corresponding to the sine transform
    i_Kzz_positive_freq = Kuu_single_block_asymmetrical_multi_dirac_spectral_kernel_inducingpoints(
        inducing_variable, kernel, kernel.means, kernel.img_powers)
    i_Kzz_negative_freq = Kuu_single_block_asymmetrical_multi_dirac_spectral_kernel_inducingpoints(
        inducing_variable, kernel, -kernel.means, kernel.img_powers)

    #NOTE -- I have to be careful in Kuf to maintain the positive, negative freq ordering
    i_Kzz = BlockDiagMat(i_Kzz_positive_freq, i_Kzz_negative_freq)

    #NOTE -- I have to be careful in Kuf to maintain the real, imaginary features ordering

    #NOTE -- this is the unwindowed case
    #r_Kzz = BlockDiagMat(tf.linalg.diag(kernel.real_powers), tf.linalg.diag(kernel.real_powers))
    #i_Kzz = BlockDiagMat(tf.linalg.diag(kernel.img_powers), tf.linalg.diag(kernel.img_powers))
    
    #NOTE -- this is the version just with positive frequencies
    #return BlockDiagMat(r_Kzz_positive_freq, i_Kzz_positive_freq)
    
    #Kzz = BlockDiagMat(r_Kzz, i_Kzz)
    Kzz = BlockDiagMat(r_Kzz_positive_freq, i_Kzz_positive_freq)

    print('Kzz')
    print(tf.shape(Kzz))

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
