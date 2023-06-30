# Copyright 2016-2020 The GPflow Contributors. All Rights Reserved.
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

from typing import NamedTuple, Optional, Tuple

import numpy as np
import tensorflow as tf
from check_shapes import check_shapes, inherit_check_shapes

from .. import posteriors
from ..base import InputData, MeanAndVariance, RegressionData, TensorData
from ..config import default_float, default_jitter
from ..covariances.dispatch import Kuf, Kuu
from ..inducing_variables import SpectralInducingPoints
from ..kernels import Kernel
from ..likelihoods import Gaussian
from ..mean_functions import MeanFunction
from ..utilities import add_noise_cov, assert_params_false, to_default_float
from .model import GPModel
from .training_mixins import InternalDataTrainingLossMixin
from .util import InducingPointsLike, data_input_to_tensor, inducingpoint_wrapper


class IFF_SGPRBase_deprecated(GPModel, InternalDataTrainingLossMixin):
    """
    Common base class for IFF-version of SGPR that provides the common __init__
    and upper_bound() methods. #NOTE -- not sure about upper_bound() in this case.
    """

    @check_shapes(
        "data[0]: [N, D]",
        "data[1]: [N, P]",
        "noise_variance: []",
    )
    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        inducing_variable: InducingPointsLike,
        *,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: Optional[int] = None,
        noise_variance: Optional[TensorData] = None,
        likelihood: Optional[Gaussian] = None,
    ):
        """
        This method only works with a Gaussian likelihood, its variance is
        initialized to `noise_variance`.

        :param data: a tuple of (X, Y), where the inputs X has shape [N, D]
            and the outputs Y has shape [N, R].
        :param inducing_variable:  an InducingPoints instance or a matrix of
            the pseudo inputs Z, of shape [M, D].
        :param kernel: An appropriate GPflow kernel object.
        :param mean_function: An appropriate GPflow mean function object.
        """
        assert (noise_variance is None) or (
            likelihood is None
        ), "Cannot set both `noise_variance` and `likelihood`."
        if likelihood is None:
            if noise_variance is None:
                noise_variance = 1.0
            likelihood = Gaussian(noise_variance)
        X_data, Y_data = data_input_to_tensor(data)
        num_latent_gps = Y_data.shape[-1] if num_latent_gps is None else num_latent_gps
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=num_latent_gps)

        self.data = X_data, Y_data
        self.num_data = X_data.shape[0]

        self.inducing_variable: SpectralInducingPoints = inducingpoint_wrapper(inducing_variable)


class IFF_SGPR_deprecated(IFF_SGPRBase_deprecated):
    """
    Sparse GP regression using IFF formulation.

    The key reference is :cite:t:`titsias2009variational`.

    For a use example see :doc:`../../../../notebooks/getting_started/large_data`.
    """

    """
    #NOTE -- don't think this is necessary
    class CommonTensors(NamedTuple):
        sigma_sq: tf.Tensor
        sigma: tf.Tensor
        A: tf.Tensor
        B: tf.Tensor
        LB: tf.Tensor
        AAT: tf.Tensor
        L: tf.Tensor
    """
        
    # type-ignore is because of changed method signature:
    @inherit_check_shapes
    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore[override]
        return self.elbo()

    @check_shapes(
        "return: []",
    )
    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """
        #TODO -- need to use the custom Talay Cheema ELBO for IFF-SGPR here
        #common = self._common_calculation()
        #output_shape = tf.shape(self.data[-1])
        #num_data = to_default_float(output_shape[0])
        #output_dim = to_default_float(output_shape[1])
        #const = -0.5 * num_data * output_dim * np.log(2 * np.pi)
        #logdet = self.logdet_term(common)
        #quad = self.quad_term(common)
        #return const + logdet + quad

        #NOTE -- corresponds to equation 15 from Appendix of IFF paper

        #TODO -- introduce self.spectrum_inducing_points to be the 
        # diagonal term of the S matrix 
        # (contains the Z_{m} evaluated at the PSD of the underlying kernel)

        log_determinant = tf.reduce_sum(tf.math.log(self.spectrum_inducing_points))
        #TODO -- need to introduce the narrow bandwidth value as self.epsilon
        log_determinant+= - self.kernel.num_inducing * self.kernel.dim_input * self.epsilon
        log_determinant+= self.num_data * tf.math.log(self.likelihood.variance)

        #NOTE -- __gamma_sq : squared L2 norm of output data
        _gamma_sq = tf.reduce_sum(tf.square(self.data[1]))
        mahalanobis_term = -tf.math.reciprocal(self.likelihood.variance) * _gamma_sq

        #NOTE -- A: 
        mahalanobis_term+= tf.math.reciprocal(tf.square(self.likelihood.variance))

        trace_term  =
        
        elbo = 

        return -0.5 * elbo



    @inherit_check_shapes
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. For a derivation of the terms in here, see the associated SGPR
        notebook.
        """
        # could copy into posterior into a fused version

        assert_params_false(self.predict_f, full_output_cov=full_output_cov)

        X_data, Y_data = self.data
        num_inducing = self.inducing_variable.num_inducing
        err = Y_data - self.mean_function(X_data)
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        Kus = Kuf(self.inducing_variable, self.kernel, Xnew)

        sigma_sq = tf.squeeze(self.likelihood.variance_at(X_data), axis=-1)
        sigma = tf.sqrt(sigma_sq)

        L = tf.linalg.cholesky(kuu)  # cache alpha, qinv
        A = tf.linalg.triangular_solve(L, kuf / sigma, lower=True)
        B = tf.linalg.matmul(A, A, transpose_b=True) + tf.eye(
            num_inducing, dtype=default_float()
        )  # cache qinv
        LB = tf.linalg.cholesky(B)  # cache alpha
        Aerr = tf.linalg.matmul(A, err / sigma[..., None])
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True)
        tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = (
                self.kernel(Xnew)
                + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
                - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
            )
            var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp2), 0)
                - tf.reduce_sum(tf.square(tmp1), 0)
            )
            var = tf.tile(var[:, None], [1, self.num_latent_gps])

        return mean + self.mean_function(Xnew), var

    @check_shapes(
        "return[0]: [M, P]",
        "return[1]: [M, M]",
    )
    def compute_qu(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Computes the mean and variance of q(u) = N(mu, cov), the variational distribution on
        inducing outputs.

        SVGP with this q(u) should predict identically to SGPR.

        :return: mu, cov
        """
        X_data, Y_data = self.data

        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())

        var = tf.squeeze(self.likelihood.variance_at(X_data), axis=-1)
        std = tf.sqrt(var)
        scaled_kuf = kuf / std
        sig = kuu + tf.matmul(scaled_kuf, scaled_kuf, transpose_b=True)
        sig_sqrt = tf.linalg.cholesky(sig)

        sig_sqrt_kuu = tf.linalg.triangular_solve(sig_sqrt, kuu)

        cov = tf.linalg.matmul(sig_sqrt_kuu, sig_sqrt_kuu, transpose_a=True)
        err = Y_data - self.mean_function(X_data)
        scaled_err = err / std[..., None]
        mu = tf.linalg.matmul(
            sig_sqrt_kuu,
            tf.linalg.triangular_solve(sig_sqrt, tf.linalg.matmul(scaled_kuf, scaled_err)),
            transpose_a=True,
        )

        return mu, cov


class IFF_SGPR_with_posterior(IFF_SGPR_deprecated):
    """
    Sparse Variational GP regression.
    The key reference is :cite:t:`titsias2009variational`.

    This is an implementation of SGPR that provides a posterior() method that
    enables caching for faster subsequent predictions.
    """

    def posterior(
        self,
        precompute_cache: posteriors.PrecomputeCacheType = posteriors.PrecomputeCacheType.TENSOR,
    ) -> posteriors.SGPRPosterior:
        """
        Create the Posterior object which contains precomputed matrices for
        faster prediction.

        precompute_cache has three settings:

        - `PrecomputeCacheType.TENSOR` (or `"tensor"`): Precomputes the cached
          quantities and stores them as tensors (which allows differentiating
          through the prediction). This is the default.
        - `PrecomputeCacheType.VARIABLE` (or `"variable"`): Precomputes the cached
          quantities and stores them as variables, which allows for updating
          their values without changing the compute graph (relevant for AOT
          compilation).
        - `PrecomputeCacheType.NOCACHE` (or `"nocache"` or `None`): Avoids
          immediate cache computation. This is useful for avoiding extraneous
          computations when you only want to call the posterior's
          `fused_predict_f` method.
        """

        return posteriors.SGPRPosterior(
            kernel=self.kernel,
            data=self.data,
            inducing_variable=self.inducing_variable,
            likelihood=self.likelihood,
            num_latent_gps=self.num_latent_gps,
            mean_function=self.mean_function,
            precompute_cache=precompute_cache,
        )

    @inherit_check_shapes
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        For backwards compatibility, GPR's predict_f uses the fused (no-cache)
        computation, which is more efficient during training.

        For faster (cached) prediction, predict directly from the posterior object, i.e.,:
            model.posterior().predict_f(Xnew, ...)
        """
        return self.posterior(posteriors.PrecomputeCacheType.NOCACHE).fused_predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )


class IFF_SGPR(IFF_SGPR_with_posterior):
    # subclassed to ensure __class__ == "SGPR"

    __doc__ = IFF_SGPR_deprecated.__doc__  # Use documentation from SGPR_deprecated.
