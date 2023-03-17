# Copyright (C) Secondmind Ltd 2021 - All Rights Reserved
# Unauthorised copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
""" This module defines primitive spectral kernels. """
import math
from statistics import variance
from typing import Any, List, Optional

import numpy as np
import tensorflow as tf
from gpflow.kernels.spectral_stationaries import IsotropicSpectralStationary, AnisotropicSpectralStationary, SpectralSum
from gpflow.kernels.base import Sum

from gpflow.kernels.spectral_utils import sinc
from gpflow.base import Parameter, TensorType
from .spectral_utils import sinc
from ..utilities.ops import difference_matrix, square_distance, batched_difference_matrix
from check_shapes import check_shapes, inherit_check_shapes
from ..utilities import positive

#NOTE -- this is a cosine kernel, which corresponds to the PSD being just a Dirac delta
class SpectralDiracDeltaBlock(AnisotropicSpectralStationary):
    """

    The primitive kernel defined by a constant spectral density spanning an infinitesimally small bandwidth (Dirac delta).
    Boils down to a cosine kernel, effectively.
    """

    def __init__(
        self,
        powers, 
        means,
        bandwidths,
        active_dims=None,
    ):
        """
        :param powers: The variance associated with the block.
        :param means: Defined as the mean frequency.
        :param bandwidths: The frequency range spanned by the spectral block.
        :param active_dims: TODO missing param description
        """
        super().__init__(powers=powers, means=means, bandwidths=bandwidths, active_dims=active_dims)

    def K_d(self, d):
        """
        #TODO -- update the documentation here
        :param d: expected_shape [N1, N2, D]

        returns: expected_shape [N1, N2]
        """

        print('----- inside K_d of  SpectralDiracDeltaBlock ------')
        cos_term = tf.cos(2 * math.pi * tf.reduce_sum(d, axis = -1)) #[N1, N2]    
        print(cos_term)

        output = self.powers * cos_term

        return output


#NOTE -- this is to be only used for the Integrated Fourier Features model 
class IFFMultipleSpectralBlock(IsotropicSpectralStationary):
    """
    
    The primitive kernel defined by multiple constant spectral density spanning finite bandwidths.
    To be used for Integrated Fourier Features models.
    Works with multi-dimensional data by taking a product over symmetrical rectangles for each input dimension.

    Important: this translates just to a squared exponential kernel to be used for Kff
    Kuu and Kuf for this type of model will be tailor made in the covariances dispatcher. 

    """

    def __init__(
        self,
        n_components: int,
        powers,
        means,
        bandwidths,
        lengthscales,
        variance,
        **kwargs: Any
    ):
        """
        :param powers: The variance associated with the block. Expected shape [M, ]
        (Corresponds to the power of the spectral block)
        #NOTE -- is it reallt defined as the inverse of the mean frequency?
        :param means: Defined as the inverse of the mean frequency. Expected shape [D, M]
        (Corresponds to the frequency location of the spectral block)
        
        :param bandwidths: The frequency range spanned by the spectral block. Expected shape [D, M]
        (Corresponds to the delta) notation
        
        :param lengthscales: TODO missing param description

        :param variance: TODO missing param description

        :param active_dims: TODO missing param description
        """

        super().__init__(powers = powers, means = means, bandwidths = bandwidths, lengthscales=lengthscales, variance=variance, **kwargs)

        self.variance = Parameter(variance, transform=positive())
        self.lengthscales = Parameter(lengthscales, transform=positive())
        self._validate_ard_active_dims(self.lengthscales)

        self.n_components = n_components

    #NOTE -- thIS gets used because the underlying kernel is a squared exponential for IFF models
    #NOTE -- this will only get used for Kff
    @inherit_check_shapes
    def K_r2(self, r2: TensorType) -> tf.Tensor:
        return self.variance * tf.exp(-0.5 * r2)

