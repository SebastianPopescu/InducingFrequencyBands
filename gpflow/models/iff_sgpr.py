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

    """
    @check_shapes(
        "return.sigma_sq: [N]",
        "return.sigma: [N]",
        "return.A: [M, N]",
        "return.B: [M, M]",
        "return.LB: [M, M]",
        "return.AAT: [M, M]",
    )
    """


    def _common_calculation(self) -> "SGPR.CommonTensors":
        """
        Matrices used in log-det calculation

        :return:
            * :math:`σ²`,
            * :math:`σ`,
            * :math:`A = L⁻¹K_{uf}/σ`, where :math:`LLᵀ = Kᵤᵤ`,
            * :math:`B = AAT+I`,
            * :math:`LB` where :math`LBLBᵀ = B`,
            * :math:`AAT = AAᵀ`,
        """
        """
        x, _ = self.data  # [N]
        iv = self.inducing_variable  # [M]

        sigma_sq = tf.squeeze(self.likelihood.variance_at(x), axis=-1)  # [N]
        sigma = tf.sqrt(sigma_sq)  # [N]

        kuf = Kuf(iv, self.kernel, x)  # [M, N]
        kuu = Kuu(iv, self.kernel, jitter=default_jitter())  # [M, M]
        L = tf.linalg.cholesky(kuu)  # [M, M]

        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, kuf / sigma, lower=True)
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = add_noise_cov(AAT, tf.cast(1.0, AAT.dtype))
        LB = tf.linalg.cholesky(B)

        return self.CommonTensors(sigma_sq, sigma, A, B, LB, AAT, L)
        """
        pass
        
    @check_shapes(
        "return: []",
    )
    def logdet_term(self, common: "SGPR.CommonTensors") -> tf.Tensor:
        r"""
        Bound from Jensen's Inequality:

        .. math::
            \log |K + σ²I| <= \log |Q + σ²I| + N * \log (1 + \textrm{tr}(K - Q)/(σ²N))

        :param common: A named tuple containing matrices that will be used
        :return: log_det, lower bound on :math:`-.5 * \textrm{output_dim} * \log |K + σ²I|`
        """
        
        """
        #NOTE -- previous version
        sigma_sq = common.sigma_sq
        LB = common.LB
        AAT = common.AAT

        x, y = self.data
        outdim = to_default_float(tf.shape(y)[1])
        kdiag = self.kernel(x, full_cov=False)

        # tr(K) / σ²
        trace_k = tf.reduce_sum(kdiag / sigma_sq)
        # tr(Q) / σ²
        trace_q = tf.reduce_sum(tf.linalg.diag_part(AAT))
        # tr(K - Q) / σ²
        trace = trace_k - trace_q

        # 0.5 * log(det(B))
        half_logdet_b = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))

        # sum log(σ²)
        log_sigma_sq = tf.reduce_sum(tf.math.log(sigma_sq))

        logdet_k = -outdim * (half_logdet_b + 0.5 * log_sigma_sq + 0.5 * trace)
        return logdet_k
        """

        x, y = self.data
        outdim = to_default_float(tf.shape(y)[1])

        iv = self.inducing_variable  # [M]
        M = iv.num_inducing

        #NOTE-- necesarry to get kernel so as to have access to frequency bands data
        kern = self.kernel
        _bandwidths = self.kernel.bandwidths

        log_det = tf.reduce_sum(tf.math.log(_bandwidths))
        A = 
        log_det+= 

        return logdet_k

    @check_shapes(
        "return: []",
    )
    def quad_term(self, common: "SGPR.CommonTensors") -> tf.Tensor:
        """
        :param common: A named tuple containing matrices that will be used
        :return: Lower bound on -.5 yᵀ(K + σ²I)⁻¹y
        """
        sigma = common.sigma
        A = common.A
        LB = common.LB

        x, y = self.data
        err = (y - self.mean_function(x)) / sigma[..., None]

        Aerr = tf.linalg.matmul(A, err)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True)

        # σ⁻² yᵀy
        err_inner_prod = tf.reduce_sum(tf.square(err))
        c_inner_prod = tf.reduce_sum(tf.square(c))

        quad = -0.5 * (err_inner_prod - c_inner_prod)
        return quad

    @check_shapes(
        "return: []",
    )
    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """
        common = self._common_calculation()
        output_shape = tf.shape(self.data[-1])
        num_data = to_default_float(output_shape[0])
        output_dim = to_default_float(output_shape[1])
        const = -0.5 * num_data * output_dim * np.log(2 * np.pi)
        logdet = self.logdet_term(common)
        quad = self.quad_term(common)
        return const + logdet + quad

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


class SGPR_with_posterior(SGPR_deprecated):
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


class SGPR(SGPR_with_posterior):
    # subclassed to ensure __class__ == "SGPR"

    __doc__ = SGPR_deprecated.__doc__  # Use documentation from SGPR_deprecated.
