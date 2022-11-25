# Copyright (C) Secondmind Ltd 2021 - All Rights Reserved
# Unauthorised copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
""" Assemble a stationary kernel from a sum of blocks or Gaussians. """
import abc
from typing import Optional, Tuple


from gpflow.kernels import Kernel
from gpflow.base import Parameter
import numpy as np
import tensorflow as tf
from gpflow.config import default_float
from pio_utilities import pio_logging
from tensorflow_probability.python.bijectors import Exp

from gpflow.utilities.bijectors import positive
from gpflow.kernels.initialisation_np import np_randomise_initial_components
from gpflow.kernels.spectral_utils import sinc

LOG = pio_logging.logger(__name__)

DEFAULT_N_COMPONENTS = 20
DEFAULT_BIAS = 0.1

# Working in log space improves optimization performance
POSITIVE_TRANSFORM = Exp()


class SpectralKernel(Kernel):
    """Compound kernels whose spectral density is defined by a sum of Gaussians or blocks."""

    # TensorFlow 2 requires jitting by wrapping in tf.function() decorator to
    # run at speed.  tf.function(autograph=True) [default] does not work with
    # gpflow.  tf.function(autograph=False) does not support converting python
    # statements (if, for) to TensorFlow ops (tf.cond(), tf.while()). As a
    # temporary workaround, the downstream AutoML _train() function disables
    # jitting when hasattr(model.kernel, "no_jit_support")
    no_jit_support = True

    # pylint: disable=arguments-differ
    @abc.abstractmethod
    def K(self, X, X2=None):
        """
        Calculate the kernel matrix ``K(X, X2)`` (or ``K(X, X)`` if ``X2`` is ``None``).

        :param X: TODO missing param description
        :param X2: TODO missing param description
        """

        pass

    # pylint: disable=arguments-differ
    @abc.abstractmethod
    def K_diag(self, X):
        """Evaluate the kernel at the provided X values."""
        pass

    def evaluate_single_component(self, r: tf.Tensor, i: int) -> tf.Tensor:
        """
        A kernel corresponding to a single Gaussian or block in the power spectrum, with unit
        variance.
        See the appendix of the following Overleaf document for further details:
        https://www.overleaf.com/read/pgjjqjfbwqjr
        Also see equations 6-11 of
        [Gaussian Process Kernels for Pattern Discovery and Extrapolation]
        (https://arxiv.org/abs/1302.4245).

        :param r: .. X D tensor or NumPy array, the unscaled separation between
            the points of interest.
        :param i: The index of the component we wish to evaluate.
        """

        cos_term = tf.cos(2 * np.pi * r * self.means[0, i])

        if self.use_blocks:
            exp_term = sinc(r * self.bandwidths[0, i])
        else:
            exp_term = tf.exp(-2 * np.pi ** 2 * tf.square(r * self.bandwidths[0, i]))

        return exp_term * cos_term

    def get_component_parameters(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Retrieve values of components."""

        return self.means, self.bandwidths, self.variances

    def set_component_parameters(self, means, bandwidths, variances):
        """Update kernel parameters."""

        self.means.assign(means)
        self.bandwidths.assign(bandwidths)
        self.variances.assign(variances)


class SpectralMixtureKernel(SpectralKernel):
    """
    A stationary kernel parameterised by the superposition of multivariate Gaussians or blocks in
    Fourier space. Currently this class is limited to operating in either one or two dimensions.
    """

    def __init__(
        self,
        input_dims: int = 1,
        active_dims=None,
        name=None,
        variances: Optional[np.ndarray] = None,
        means: Optional[np.ndarray] = None,
        bandwidths: Optional[np.ndarray] = None,
        n_components: int = DEFAULT_N_COMPONENTS,
        use_blocks: bool = False,
        use_bias: bool = False,
    ):
        """
        :param input_dims: Currently only one-dimensional data is supported.
        :param active_dims: TODO add param description
        :param name: TODO add param description
        :param variances: (Optional) The relative amplitude of each component, of shape ``DxN``,
            where ``N`` is the number of components.
        :param means: (Optional) The mean frequency of each component, of shape ``DxN``, where
            ``N`` is the number of components.
        :param bandwidths: (Optional) The variance of each component within the power spectrum, the
            breadth of frequencies it spans.
        :param n_components: How many spectral components are used in the model.
        :param use_blocks: Whether to model the spectral density with blocks or Gaussians.
        :param use_bias: Whether to include a bias kernel (constant offset).
        """

        assert input_dims == 1, "Currently only 1-dimensional data is supported."

        super().__init__(active_dims, name=name)

        self.input_dim = input_dims
        self.use_blocks = use_blocks
        if variances is None:
            means, bandwidths, variances = self._get_default_params(input_dims, n_components)
            self.n_components = n_components
        else:
            self.n_components = np.size(variances)

        self.bandwidths = Parameter(
            bandwidths,
            dtype=default_float(),
            transform=POSITIVE_TRANSFORM,
            trainable=True,
            name="bandwidths",
        )
        self.means = Parameter(
            np.abs(means),
            transform=POSITIVE_TRANSFORM,
            dtype=default_float(),
            trainable=True,
            name="means",
        )
        self.variances = Parameter(
            tf.abs(variances),
            transform=POSITIVE_TRANSFORM,
            dtype=default_float(),
            trainable=True,
            name="variances",
        )

        if use_bias:
            self.bias = Parameter(
                DEFAULT_BIAS,
                transform=positive(),
                dtype=default_float(),
                trainable=use_bias,
                name="bias",
            )
        else:
            self.bias = tf.constant(0.0, dtype=default_float())

    @property
    def variance(self):
        """The total variance of the kernel is the sum of its spectral components."""
        return tf.reduce_sum(input_tensor=self.variances) + self.bias

    def K(self, X: tf.Tensor, X2: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Calculate the kernel matrix ``K(X, X2)`` (or ``K(X, X)`` if ``X2`` is `None`).
        This function handles the slicing as well as the scaling, and computes ``k(x, x') = k(r)``,
        where ``r = x - x'``.

        :param X: TODO add missing parameter description
        :param X2: TODO add missing parameter description
        """

        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting to compute all pairwise separations
        f = tf.expand_dims(X, -2)  # ... x N x 1 x D
        f2 = tf.expand_dims(X2, -3)  # ... x 1 x M x D

        r = f - f2  # N x M x D
        original_shape = r.shape
        reshaped_r = tf.reshape(r, (-1, self.input_dim))

        output = self.K_r(reshaped_r)
        output = tf.reshape(output, original_shape[:2])

        return output

    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        """Evaluate the kernel at the provided X values."""
        return tf.fill(X.shape[:-1], tf.squeeze(self.variance))

    def K_r(self, r: tf.Tensor) -> tf.Tensor:
        """
        Evaluate the kernel as a function of separation ``r``.
        .. note:: This ``r`` is not rescaled as it is in native GPflow kernels.

        :param r: TODO add missing parameter description
        """
        return self.evaluate_correlation_function(r)

    # FIXME: this function should not need to be compiled as it should already be inside
    #  a tf.function. The error message suggests it's to do with the iteration
    #  (may or may not be the one inside the present function):
    #  `OperatorNotAllowedInGraphError: iterating over `tf.Tensor` is not allowed:
    #  AutoGraph did not convert this function. Try decorating it directly with @tf.function.`
    #  We should dig deeper when we have the time.
    @tf.function
    def evaluate_correlation_function(self, r: tf.Tensor) -> tf.Tensor:
        """
        Sum the spectral components, yielding a correlation function of unit variance ``(K(r=0)=1)``
        which can then be amplified by ``self.variance``. However, this does not include the
        contribution from likelihood variance.

        :param r: An NxD tensor of separation values.
        """
        xi_shape = r.shape

        cumulative_xi = tf.zeros(xi_shape, dtype=default_float())

        for i in tf.range(self.n_components):
            cumulative_xi += self.variances[i] * self.evaluate_single_component(r, i)

        return cumulative_xi + self.bias

    @staticmethod
    def _get_default_params(
        input_dims: int, n_components: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Provide some valid ``gpflow.Parameter`` values if they had not been specified.

        :param input_dims: The dimensionality of the input.
        :param n_components: TODO add missing parameter description
        :return: TODO add missing return statement
        """
        nyquist_freqs = np.ones(input_dims) * 0.5
        means, bandwidths, variances = np_randomise_initial_components(
            nyquist_freqs, n_components, 64.0
        )
        return means, bandwidths, variances

    def get_spectral_component_amplitudes(self, *args):
        """A helper function for the nonstationary spectral kernels."""

        return tf.sqrt(self.variances)
