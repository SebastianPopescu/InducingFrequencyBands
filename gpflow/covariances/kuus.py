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

def midpoint_rule(x, std, a, b, n):

    # Calculate the width of each subinterval
    h = (b - a) / n
    
    # Calculate the midpoint values for each subinterval
    mean_midpoints = tf.linspace(a + h/2, b - h/2, n)
    
    # Evaluate the function at the midpoint values

    #NOTE -- this is usign the Burmann series second order approximation
    #f_midpoints = burmann_series_approx_erf(x, mean_midpoints, std)
    #NOTE -- this is using the default Tensorflow version
    f_midpoints = tf.math.erf((x-mean_midpoints)/(tf.cast(tf.math.sqrt(2.),
                                                               default_float())*std))

    # Calculate the approximate integral using the midpoint rule formula
    integral_approx = h * tf.reduce_sum(f_midpoints)
    
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

@Kuu.register(SpectralInducingVariables, MultipleSpectralBlock)
def Kuu_block_multi_spectral_kernel_inducingpoints(
    inducing_variable: SpectralInducingVariables, kernel: MultipleSpectralBlock, *, jitter: float = 0.0
) -> tf.Tensor:
    
    #NOTE -- this is the unwindowed case
    #Kzz = tf.linalg.diag(kernel.powers)
    #Kzz = tf.cast(Kzz, default_float())

    #NOTE -- implements the following integral but approximated numerically
    #NOTE -- erf is implemented via the second order Burmann series approximation
    #TODO -- do I need  to use tf.stop_gradients here?

    num_approx_N = 1000

    Kzz = tf.linalg.diag(kernel.powers / (2. * kernel.bandwidths))

    lower_limit = kernel.means - 0.5 * kernel.bandwidths
    upper_limit = kernel.means + 0.5 * kernel.bandwidths
    
    num_int_approx = midpoint_rule(upper_limit, std = tf.cast(tf.math.sqrt(kernel.alpha)
                                                              , default_float()) / 
                                   tf.cast(np.pi, default_float()), 
                                   a = lower_limit, b = upper_limit, n = num_approx_N)
    
    num_int_approx -= midpoint_rule(lower_limit, std = tf.cast(tf.math.sqrt(kernel.alpha)
                                                               , default_float()) / 
                                    tf.cast(np.pi,default_float()), 
                                   a = lower_limit, b = upper_limit, n = num_approx_N)
    
    Kzz *= tf.linalg.diag(num_int_approx)

    print('--- inside Kzz ----')
    print(Kzz)

    #TODO -- fix this little hack, the midpoint rule is giving me a leading 1 dimension
    return tf.squeeze(2. * Kzz, axis=0)

    
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
