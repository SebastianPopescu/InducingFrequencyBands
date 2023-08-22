# Copyright (C) Secondmind Ltd 2021 - All Rights Reserved
# Unauthorised copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
""" This module defines primitive spectral kernels. """
import math
from statistics import variance
from typing import Any, List, Optional

import numpy as np
import tensorflow as tf
from gpflow.kernels.spectral_stationaries import AnisotropicSpectralStationary, SpectralSum, DecomposedAnisotropicSpectralStationary
from gpflow.kernels.base import Sum

from gpflow.kernels.spectral_utils import sinc
from gpflow.base import Parameter, TensorType
from .spectral_utils import sinc

from ..utilities.ops import difference_matrix, square_distance, batched_difference_matrix


class SpectralBlock(AnisotropicSpectralStationary):
    """
    The primitive kernel defined by a constant spectral density spanning a finite bandwidth.
    This is essentially used for the GP-SINC model introduced in Tobar (2019) [1].
    Works with multi-dimensional data by taking a product over symmetrical rectangles 
    for each input dimension.
    
    [1]: Tobar F. Band-limited Gaussian processes: The sinc kernel. 
    Advances in Neural Information Processing Systems. 2019; 32.
    """

    def __init__(
        self,
        powers, 
        means,
        bandwidths,
        alpha,
        active_dims=None,
    ):
        """
        :param powers: The variance/magnitude associated with the block.
        :param means: Defined as the mean frequency.
        :param bandwidths: The frequency range spanned by the spectral block.
        :param active_dims: TODO missing param description
        """
        super().__init__(powers=powers, means=means, bandwidths=bandwidths, 
                         alpha = alpha, active_dims=active_dims)

    def K_d(self, d):
        """
        
        Implements anisotropic version of Sinc kernel.

        Computes:
            SK(d) = σ2 sinc(∆t) cos(2πξd)
            where:
             σ2 - magnitude
             ∆ - bandwidth
             ξ - frequency
            
        :param d: expected_shape [N1, N2, D]
        returns: expected_shape [N1,N2]
        """
        cos_term = tf.cos(2 * math.pi * tf.reduce_sum(d, axis = -1)) #[N1, N2]    

        pre_multiplier = tf.transpose(self.bandwidths / self.means) # [1, D]

        sinc_term = tf.reduce_prod(
            sinc(
            tf.multiply( d, #[N1, N2, D]
            pre_multiplier[None, :] # [1, 1, D]
        ))
        , axis = -1 ) # [N1, N2]

        output = self.powers * cos_term * sinc_term

        return output

class SpectralDiracDeltaBlock(AnisotropicSpectralStationary):
    """

    The primitive kernel defined by a constant spectral density spanning an infinitesimally 
    small bandwidth (Dirac delta), essentially transforms into a cosine kernel.

    """

    def __init__(
        self,
        powers, 
        means,
        bandwidths,
        alpha,
        active_dims=None,
    ):
        """
        :param powers: The variance associated with the block.
        :param means: Defined as the mean frequency.
        :param bandwidths: The frequency range spanned by the spectral block.
        :param active_dims: TODO missing param description
        """
        super().__init__(powers=powers, means=means, bandwidths=bandwidths, 
                         active_dims=active_dims, alpha=alpha)

    def K_d(self, d):
        """
        #TODO -- update the documentation here
        :param d: expected_shape [N1, N2, D]

        returns: expected_shape [N1,N2]
        
        Remainder: because of scale function, in our implementation d is already multipled by frequency/means 
        """

        cos_term = tf.cos(2 * math.pi * tf.reduce_sum(d, axis = -1)) #[N1, N2]    
        output = self.powers * cos_term

        return output

class MultipleSpectralBlock(AnisotropicSpectralStationary):

    """
    
    The primitive kernel defined by multiple constant spectral density spanning finite bandwidths.
    To be used for Inducing Frequency Bands models and gp-multiSinc.
    Works with multi-dimensional data by taking a product over symmetrical rectangles for each input dimension.

    """

    def __init__(
        self,
        n_components: int,
        powers,
        means,
        alpha,
        bandwidths,
        **kwargs: Any
    ):
        """
        :param powers: The variance associated with the block. Expected shape [M, ]
        (Corresponds to the power of the spectral block)
        
        :param means: Defined as the inverse of the mean frequency. Expected shape [D, M]
        (Corresponds to the frequency location of the spectral block)
        
        :param bandwidths: The frequency range spanned by the spectral block. Expected shape [D, M]
        (Corresponds to the delta) notation
        
        :param alpha: TODO missing param description

        :param active_dims: TODO missing param description
        """

        super().__init__(powers = powers, means = means, bandwidths = bandwidths, 
                         alpha=alpha, **kwargs)

        self.n_components = n_components

    #TODO -- re-introduce check_shapes
    #@check_shapes(
    #    "X: [broadcast any...]",
    #    "return: [any...]",
    #)
    def scale(self, X: TensorType) -> TensorType:
        
        """
        Overwritting the default method from SpectralKernel.
        
        In the case of Inducing Frequency Bands or GP-MultiSinc
        :param X: expected shape [N, D]
        :param self.means: expected shape [D, M]
        
        :return: expected shape [N, D, M]
        """

        X_scaled = tf.expand_dims(X, axis =-1) * tf.expand_dims(self.means, axis = 0)

        return X_scaled


    #TODO -- reintroduce check_shapes
    #@check_shapes(
    #    "X: [batch..., N, D]",
    #    "X2: [batch2..., N2, D]",
    #    "return: [batch..., N, batch2..., N2, D] if X2 is not None",
    #    "return: [batch..., N, N, D] if X2 is None",
    #)
    def scaled_difference_matrix(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        """
        
        We are overriding the default method from AnisotropicSpectralStationary 
        just for the Inducing Frequency Bands project or GP-MultiSinc.
        
        #TODO -- update the documentation here
        Returns [(X - X2ᵀ) / ℓ]. If X has shape [..., N, D] and
        X2 has shape [..., M, D], the output will have shape [..., N, M, D].
        """
        if X2 is None:
            X2 = X
        
        diff_matrix = batched_difference_matrix(tf.transpose(self.scale(X), [2, 0, 1]), # expected shape [M, N1, D] 
            tf.transpose(self.scale(X2), [2, 0, 1]) # expected shape [M, N2, D] 
            ) # expected shape [M, N1, N2, D]

        return diff_matrix


    def K_d(self, d):
        """

        :param d: expected_shape [M, N1, N2, D]
       
        Implements anisotropic version of MultiSinc kernel 
        (i.e., multi-block variant of Tobar's SincGP paper).

        Computes the multi-block variant of this:
            SK(d) = σ2 sinc(∆t) cos(2πξd)
            where:
             σ2 - magnitude
             ∆ - bandwidth
             ξ - frequency
        Remainder: because of scale function, in our implementation d is already multipled by frequency/means             

        returns: expected_shape [N1,N2]
        """

        #TODO -- shouldn't this be a reduce_prod last dim outside tf.cos???
        cos_term = tf.cos(2 * math.pi * tf.reduce_sum(d, axis = -1)) #[M, N1, N2]
    
        pre_multiplier = tf.transpose(self.bandwidths / self.means) # [M, D]
        
        sinc_term = tf.reduce_prod(
            sinc(tf.multiply( d, #[M, N1, N2, D]
            pre_multiplier[:, None, None, :] # [M, 1, 1, D]
        ))
        , axis = -1 ) # [M, N1, N2]

        output = self.powers[:,None, None] * cos_term * sinc_term
    
        return tf.reduce_sum(output, axis = 0)


class DecomposedMultipleSpectralBlock(DecomposedAnisotropicSpectralStationary):

    """
    TODO -- update the documentation.
    The primitive kernel defined by multiple constant spectral density spanning finite bandwidths.
    To be used for Inducing Frequency Bands models and gp-multiSinc.
    Works with multi-dimensional data by taking a product over symmetrical rectangles for each input dimension.

    """

    def __init__(
        self,
        n_components: int,
        real_powers,
        img_powers,
        means,
        alpha,
        bandwidths,
        **kwargs: Any
    ):
        """
        #TODO -- update documentation.
        :param real_powers: The variance associated with the block. Expected shape [M, ]
        (Corresponds to the power of the spectral block)
        
        :param img_powers: The variance associated with the block. Expected shape [M, ]
        (Corresponds to the power of the spectral block)

        :param means: Defined as the inverse of the mean frequency. Expected shape [D, M]
        (Corresponds to the frequency location of the spectral block)
        
        :param bandwidths: The frequency range spanned by the spectral block. Expected shape [D, M]
        (Corresponds to the delta) notation
        
        :param alpha: TODO missing param description

        :param active_dims: TODO missing param description
        """

        super().__init__(real_powers = real_powers, img_powers = img_powers, 
                         means = means, bandwidths = bandwidths, 
                         alpha=alpha, **kwargs)

        self.n_components = n_components

    #TODO -- re-introduce check_shapes
    #@check_shapes(
    #    "X: [broadcast any...]",
    #    "return: [any...]",
    #)
    def scale(self, X: TensorType) -> TensorType:
        
        """
        Overwritting the default method from SpectralKernel.
        
        In the case of Inducing Frequency Bands or GP-MultiSinc
        :param X: expected shape [N, D]
        :param self.means: expected shape [D, M]
        
        :return: expected shape [N, D, M]
        """

        X_scaled = tf.expand_dims(X, axis =-1) * tf.expand_dims(self.means, axis = 0)

        return X_scaled


    #TODO -- reintroduce check_shapes
    #@check_shapes(
    #    "X: [batch..., N, D]",
    #    "X2: [batch2..., N2, D]",
    #    "return: [batch..., N, batch2..., N2, D] if X2 is not None",
    #    "return: [batch..., N, N, D] if X2 is None",
    #)
    def scaled_difference_matrix(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        """
        
        We are overriding the default method from AnisotropicSpectralStationary 
        just for the Inducing Frequency Bands project or GP-MultiSinc.
        
        #TODO -- update the documentation here
        Returns [(X - X2ᵀ) / ℓ]. If X has shape [..., N, D] and
        X2 has shape [..., M, D], the output will have shape [..., N, M, D].
        """
        if X2 is None:
            X2 = X
        
        diff_matrix = batched_difference_matrix(tf.transpose(self.scale(X), [2, 0, 1]), # expected shape [M, N1, D] 
            tf.transpose(self.scale(X2), [2, 0, 1]) # expected shape [M, N2, D] 
            ) # expected shape [M, N1, N2, D]

        return diff_matrix


    def K_d(self, d):
        """

        :param d: expected_shape [M, N1, N2, D]
       
        Implements anisotropic version of MultiSinc kernel 
        (i.e., multi-block variant of Tobar's SincGP paper).

        Computes the multi-block variant of this:
            SK(d) = σ2 sinc(∆t) cos(2πξd)
            where:
             σ2 - magnitude
             ∆ - bandwidth
             ξ - frequency
        Remainder: because of scale function, in our implementation d is already multipled by frequency/means             

        returns: expected_shape [N1,N2]
        """

        #TODO -- shouldn't this be a reduce_prod last dim outside tf.cos???
        cos_term = tf.cos(2 * math.pi * tf.reduce_sum(d, axis = -1)) #[M, N1, N2]
    
        pre_multiplier = tf.transpose(self.bandwidths / self.means) # [M, D]
        
        sinc_term = tf.reduce_prod(
            sinc(tf.multiply( d, #[M, N1, N2, D]
            pre_multiplier[:, None, None, :] # [M, 1, 1, D]
        ))
        , axis = -1 ) # [M, N1, N2]

        output = (self.real_powers[:, None, None] + self.img_powers[:, None, None]) * cos_term * sinc_term
    
        return tf.reduce_sum(output, axis = 0)




class SpectralGaussian(AnisotropicSpectralStationary):
    """The primitive kernel whose spectral density is comprised of a 
    pair of symmetrical Gaussians."""

    def __init__(self, 
        powers, # stands for A -- expected shape [1, ]
        means, # stands for mu -- expected shape [D, 1]
        bandwidths, # stands for sigma -- expected shape [D, 1]
        alpha,
        active_dims=None):

        super().__init__(powers=powers, 
            means=means,
            bandwidths=bandwidths, 
            alpha=alpha,
            active_dims=active_dims)

    def K_d(self, d):
        """
        The covariance associated with a spectral density given by a Gaussian.
        :param d: expected shape [N1, N2, D]

        #TODO -- update the documentation here
        :param d: The scaled Euclidean distance, which is why the lengthscale term appears inaxis
            the exponential term rather than the cosine term.
            For further details, see equations 6-11 of
            [Gaussian Process Kernels for Pattern Discovery and Extrapolation]
            (https://arxiv.org/abs/1302.4245).
        """

        cos_term = tf.cos(2 * math.pi * tf.reduce_sum(d, axis = -1)) # expected shape -- [N1, N2]
       
        descaled_d = d * tf.transpose(tf.math.reciprocal(self.means))[None, :, :] # expected shape [N1, N2, D]
        exponential_term = tf.exp(-
                                  tf.reduce_sum(tf.square(descaled_d) * 
                                                tf.transpose(self.bandwidths)[None, :, :]
                                                , axis=-1))

        return self.powers * cos_term * exponential_term

#TODO -- this is sort of breaking the encapsulation in the dispatch methods in covariances
def MixtureSpectralGaussian(
    n_components: int,
    means,
    bandwidths,
    powers,
    alpha,
) -> Sum:
    """
    A helper function to combine several spectral Gaussian kernels.

    :param n_components: How many blocks to combine.
    :param means: (Optional) The mean frequencies, of the dimensions ``DxN``, where ``N`` is
        the number of components, and ``D`` is the number of dimensions.
    :param bandwidths: (Optional) How broad each spectral block is.
    :param variances: (Optional) The variances of each block.
    :return: TODO missing return statement
    """

    #assert (means is None and bandwidths is None and variances is None) or (
    #    means is not None and bandwidths is not None and variances is not None
    #), "Either all three spectral parameters should be provided, or none"


    make_primitive_kernel = SpectralGaussian

    list_kernels = []
    for i in range(n_components):

        if means is None:
            basis_kernel = make_primitive_kernel()
        else:
            power = powers[i]
            mean = means[:,i].reshape(-1,1)
            bandwidth = bandwidths[:,i].reshape(-1,1)

            basis_kernel = make_primitive_kernel(power, mean, bandwidth, alpha)

        list_kernels.append(basis_kernel)

    sum_kernel = SpectralSum(list_kernels)

    return sum_kernel


class MixtureSpectralGaussianVectorized(AnisotropicSpectralStationary):
    """spectral density is comprised of 
    pairs of symmetrical Gaussians.
    
    To be used only with BNSE model.
    """

    def __init__(self, 
        powers, # stands for A -- expected shape [Q, ]
        means, # stands for mu -- expected shape [D, Q]
        bandwidths, # stands for sigma -- expected shape [D, Q]
        alpha,
        active_dims=None):

        super().__init__(powers=powers, 
            means=means,
            bandwidths=bandwidths, 
            alpha=alpha,
            active_dims=active_dims)

    #NOTE -- overriding inherited function from ``AnisotropicSpectralStationary''
    def scaled_difference_matrix(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        """
        Returns [(X - X2ᵀ) / ℓ]. If X has shape [..., N, D] and
        X2 has shape [..., M, D], the output will have shape [..., N, M, D].
        """
        X_scaled = tf.transpose(self.scale(X), [2,0,1]) # [Q, N1, D]

        if X2 is None:
            X2 = X
        X2_scaled = tf.transpose(self.scale(X2), [2,0,1]) # [Q, N2, D]


        diff = batched_difference_matrix(X_scaled, X2_scaled) # [Q, N1, N2, D]


        return diff
        

    #NOTE -- overriding inherited function from ``SpectralStationary''
    def scale(self, X: TensorType) -> TensorType:
        
        """
        :param X: expected shape [N, D]
        :param self.means: expected shape [D, Q], Q -- are the number of mixtures for the Mixture Spectral Kernel.
        :return: expected shape [N, D, Q]
        """

        X_scaled = X[...,tf.newaxis] * self.means[tf.newaxis,...] if X is not None else X # [N, D, Q]

        return X_scaled

    def K_d(self, d):
        """
        The covariance associated with a spectral density given by a Gaussian.
        :param d: expected shape [Q, N1, N2, D], where
        Q is the number of mixtures for the Mixture Spectral Kernel.

        #TODO -- update the documentation here
        :param d: The scaled Euclidean distance, which is why the lengthscale term appears inaxis
            the exponential term rather than the cosine term.
            For further details, see equations 6-11 of
            [Gaussian Process Kernels for Pattern Discovery and Extrapolation]
            (https://arxiv.org/abs/1302.4245).
        """

        cos_term = tf.cos(2 * math.pi * tf.reduce_sum(d, axis = -1)) # expected shape -- [Q, N1, N2]

        # means - [D, Q]       
        descaled_d = d * tf.transpose(tf.math.reciprocal(self.means)
                                      )[:, tf.newaxis, tf.newaxis, :] # expected shape [Q, N1, N2, D]

        # bandwidths - [D, Q]
        exponential_term = tf.exp(-
                                  tf.reduce_sum(tf.square(descaled_d) * 
                                                tf.transpose(self.bandwidths)[:, tf.newaxis, tf.newaxis, :]
                                                , axis=-1)) # expected shape [Q, N1, N2]

        # powers - [Q, ]
        return tf.reduce_sum(self.powers[:, tf.newaxis, tf.newaxis]**2 * cos_term * exponential_term, axis = 0) # expected shape [N1, N2]