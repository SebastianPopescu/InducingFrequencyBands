
from typing import Optional

import numpy as np
import tensorflow as tf
from check_shapes import check_shapes, inherit_check_shapes

from .. import kullback_leiblers, posteriors
from ..base import AnyNDArray, InputData, MeanAndVariance, Parameter, RegressionData
from ..conditionals import conditional
from ..config import default_float
from ..inducing_variables import InducingVariables
from ..kernels import Kernel
from ..likelihoods import Likelihood
from ..mean_functions import MeanFunction
from ..utilities import positive, triangular
from .model import GPModel
from .training_mixins import ExternalDataTrainingLossMixin
from .util import InducingVariablesLike, inducingpoint_wrapper
from ..utilities.ops import difference_matrix, batched_difference_matrix

from .. import posteriors
from ..base import InputData, MeanAndVariance, RegressionData, TensorData
from ..kernels import Kernel, MixtureSpectralGaussianVectorized
from ..likelihoods import Gaussian
from ..logdensities import multivariate_normal
from ..mean_functions import MeanFunction
from ..utilities import add_likelihood_noise_cov, assert_params_false
from .model import TimeSpectrumGPModel
from .training_mixins import InternalDataTrainingLossMixin
from .util import data_input_to_tensor


class BNSE(TimeSpectrumGPModel, InternalDataTrainingLossMixin):

    """
    This is the Bayesian Nonparametric Spectrum Estimation.

    The key reference is :cite:t:`tobar201?`.
    """

    @check_shapes(
        "data[0]: [N, D]",
        "data[1]: [N, P]",
        "noise_variance: []",
    )
    def __init__(
        self,
        data: RegressionData,
        mean_function: Optional[MeanFunction] = None,
        noise_variance: Optional[TensorData] = None,
        likelihood: Optional[Gaussian] = None,
        aim = None,
    ):
        assert (noise_variance is None) or (
            likelihood is None
        ), "Cannot set both `noise_variance` and `likelihood`."
        _, Y_data = data

        X,Y = data
        self.x = X 
        self.y = Y

        self.initialise_kernel_hyperparams()
        if likelihood is None:
            if noise_variance is None:
                noise_variance = self.sigma_n**2
            likelihood = Gaussian(noise_variance)
        
        kernel = self.set_kernel()
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1])
        self.data = data_input_to_tensor(data)

        if aim is None:
            pass

        elif aim == 'sampling':
            """
            For sampling ..
            """
            #TODO -- see if we can replicate the plots from the main paper.
            self.sigma = 1
            self.gamma = 0.5
            self.theta = 0
            self.sigma_n = 0

        elif aim == 'regression':
            """
            For regression ...
            """
            self.Nx = len(self.x)

    def set_kernel(self):

        return MixtureSpectralGaussianVectorized(
                                       means = self.theta,
                                       bandwidths = self.gamma,
                                       powers = self.sigma,
                                       alpha = self.alpha)

    def initialise_kernel_hyperparams(self):
    
        #NOTE -- is Tobar taking just one Gaussian for the Mixture Spectral Kernel?

        self.Nx = len(self.x) # number of observations
        self.alpha = 0.5 / ((np.max(self.x) - 
                             np.min(self.x))
                             /2.)**2 # to be used for windowing function 
        self.sigma = [np.std(self.y)] 
        
        self.gamma = 0.5 / ((np.max(self.x) - 
                             np.min(self.x))
                             /self.Nx)**2
        self.gamma = np.reshape(self.gamma, (1,-1))

        self.theta = 0.01
        self.theta = np.reshape(self.theta, (1,-1))

        self.sigma_n = np.std(self.y) / 10.
        
        #TODO -- maybe just for notebooks?
        #self.time = np.linspace(np.min(self.x), np.max(self.x), 500)
        #self.w = np.linspace(0, self.Nx / (np.max(self.x) - 
        #                                   np.min(self.x)) / 16., 500)

    # type-ignore is because of changed method signature:
    @inherit_check_shapes
    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore[override]
        return self.log_marginal_likelihood()

    @check_shapes(
        "return: []",
    )
    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        X, Y = self.data
        K = self.kernel(X)
        ks = add_likelihood_noise_cov(K, self.likelihood, X)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(X)

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = multivariate_normal(Y, m, L)
        return tf.reduce_sum(log_prob)
                
    def conditional_GP_equations(self, Kuu, Kuf, Kff, Y, full_cov):

        """
        Standard GP conditional equations for either 
        time-domain or frequency-domain predictions.
        """

        Lm = tf.linalg.cholesky(Kuu)

        A = tf.linalg.triangular_solve(Lm, Kuf, lower=True)  # [..., M, N]
        if full_cov:
            fvar = Kff - tf.linalg.matmul(A, A, transpose_a=True)  # [..., N, N]
        else:
            fvar = Kff - tf.reduce_sum(tf.square(A), -2)  # [..., N]
            #cov_shape = tf.concat([leading_dims, [num_func, N]], 0)  # [..., R, N]
            #fvar = tf.broadcast_to(tf.expand_dims(fvar, -2), cov_shape)  # [..., R, N]

        A = tf.linalg.triangular_solve(tf.linalg.adjoint(Lm), A, lower=False)
        fmean = tf.linalg.matmul(A, Y, transpose_a=True)  # [..., N, R]

        return fmean, fvar

    def compute_moments_Xnew(self, Xnew, X, Y, full_cov = False):
        
        """
        Standard GP conditional equations in the time domain.
        """

        #Kmm = self.Spec_Mix(X, X) + 1e-5*tf.eye(self.Nx, dtype = default_float()) + self.sigma_n**2 * tf.eye(self.Nx, dtype = default_float())
        #Knn = self.Spec_Mix(Xnew, Xnew)
        #Kmn = self.Spec_Mix(X, Xnew)

        #TODO -- use the added jitter function native to GPflow
        Kmm = self.kernel(X, X) + 1e-5*tf.eye(self.Nx, dtype = default_float())
        Kmm = add_likelihood_noise_cov(K = Kmm, likelihood = self.likelihood, X = X)
        Knn = self.kernel(Xnew, Xnew)
        Kmn = self.kernel(X, Xnew)

        fmean, fvar = self.conditional_GP_equations(Kmm, Kmn, Knn, Y, full_cov)

        return fmean, fvar

    def compute_moments(self, Xnew, full_cov = False):

        ### posterior moments for time domain ###
        self.post_mean, self.post_cov = self.compute_moments_Xnew(Xnew, self.x, self.y, full_cov=full_cov)
    
        ### posterior moments for frequency domain ###
        cov_real, cov_imag = self.complex_gp_spectrum_covariances(self.w, self.w,  kernel = 'sm')
        xcov_real, xcov_imag = self.time_freq_covariances(self.w, self.x, kernel = 'sm')

        #TODO -- use the added jitter function native to GPflows
        Kmm = self.kernel(self.x, self.x) + 1e-5*tf.eye(self.Nx, dtype = default_float())
        Kmm = add_likelihood_noise_cov(K = Kmm, likelihood = self.likelihood, X = self.x)
        #Kmm = self.Spec_Mix(X, X) + 1e-5*tf.eye(self.Nx, dtype = default_float()) + self.sigma_n**2 * tf.eye(self.Nx, dtype = default_float())

        self.post_mean_r, self.post_cov_r = self.conditional_GP_equations(Kmm, 
                                                                          tf.transpose(xcov_real), 
                                                                          cov_real, 
                                                                          self.y, 
                                                                          full_cov=full_cov)

        self.post_mean_i, self.post_cov_i = self.conditional_GP_equations(Kmm, 
                                                                          tf.transpose(xcov_imag), 
                                                                          cov_imag, 
                                                                          self.y, 
                                                                          full_cov=full_cov)
        #TODO -- maybe this can be written somewhere else
        Lm = tf.linalg.cholesky(Kmm)
        A_right = tf.linalg.triangular_solve(Lm, tf.transpose(xcov_imag), lower=True)  # [..., ?,?]
        A_left = tf.linalg.triangular_solve(Lm, tf.transpose(xcov_real), lower=True)  # [..., ?,?]
        
        #TODO -- I am not sure this is okay
        self.post_cov_ri = - tf.linalg.matmul(A_left, A_right, transpose_a=True)  # [..., ?,?]
        #post_cov_ri = - ((xcov_real@np.linalg.solve(Kmm, xcov_imag.T)))
        
        self.post_mean_F = tf.concat( [self.post_mean_r, self.post_mean_i], axis = -1) 
        self.post_cov_F = tf.experimental.numpy.vstack(
            (tf.experimental.numpy.hstack((self.post_cov_r, self.post_cov_ri)), 
             tf.experimental.numpy.hstack((tf.transpose(self.post_cov_ri), self.post_cov_i))
             )) 
        
        return cov_real, xcov_real, Kmm
    
 


    def complex_gp_spectrum_covariances(self, x, y, kernel = 'sm'):

        r"""
        Local Spectrum \mathcal{F}_{c}\left(  \xi \right) 
        is a complex GP, meaning it posses both a covariance
        and a pseudo-covariance.

        #TODO -- find eqns in paper.
        K and P correspond to equations ...

        Real and Imaginary covariances correspond to equations ... 
        """
        
        if kernel == 'sm':
            N = len(x)

            # Spectrum covariance
            # 0.5 scaling is due to missing 0.5 scaling in ``spectrum_covariance''          
            K = 0.5*(self.spectrum_covariance(x, y, self.kernel.means) + 
                    self.spectrum_covariance(x, y, -self.kernel.means))
            # Spectrum pseudo-covariance
            P = 0.5*(self.spectrum_covariance(x, -y, self.kernel.means) + 
                    self.spectrum_covariance(x, -y, -self.kernel.means))
            # Krr -- real covariance
            real_cov = 0.5*(K + P) + 1e-8*tf.eye(N, dtype = default_float())
            # Kii -- imaginary covariance
            imag_cov = 0.5*(K - P) + 1e-8*tf.eye(N, dtype = default_float())
            #NOTE -- remainder: Kir = Kri = 0 since the underlying signal is real-valued.

        return real_cov, imag_cov



    def spectrum_covariance(self, xi1, xi2, theta):

        r"""
        Computes K_{ff}\left( \xi, \xi' \right).

        Corresponds to equation 17 from BNSE paper.

        xi1 # [M,]
        xi2 # [M',]
        """

        """
        magnitude = np.pi * sigma**2 / (np.sqrt(alpha*(alpha + 2*gamma)))
        return magnitude * np.exp(-np.pi**2/(2*alpha)*outersum(x,-y)**2 - 2*np.pi*2/(alpha + 2*gamma)*(outersum(x,y)/2-theta)**2)
        """

        #NOTE -- this will only work for 1D data!
        _alpha = tf.reshape(self.kernel.alpha, [-1,]) # [1,]
        _gamma = tf.reshape(self.kernel.bandwidths, [-1,]) # [Q,]
        _sigma = tf.reshape(self.kernel.powers, [-1,]) # [Q,]
        _theta = tf.reshape(theta, [-1,]) # [Q,]

        _pi = np.pi

        magnitude = _pi * _sigma**2 / (tf.sqrt(_alpha * (_alpha + 2.*_gamma))) # [Q,]
        magnitude = magnitude[tf.newaxis, :, tf.newaxis] # [1, Q, 1]
 
        Kxi_xi = tf.math.exp(-_pi**2/(2.*_alpha[tf.newaxis, :, tf.newaxis]) * #  [1, 1, 1]
                             tf.square(outersum(xi1,-xi2))[:,tf.newaxis,:] # [M, 1, M']
                             ) # [M, 1, M']
        Kxi_xi *= tf.math.exp( - 2.*_pi**2/
                              (_alpha + 2.*_gamma)[tf.newaxis, :, tf.newaxis] # [1, Q, 1]
                              * tf.square(outersum(xi1,xi2)[:,tf.newaxis,:]/2. # [M, 1, M']
                                          -tf.reshape(_theta,[1,-1,1]) # [1, Q, 1]
                                          ) # [M, Q, M']
                              ) # [M, Q, M']
        
        return tf.reduce_sum(magnitude * Kxi_xi, axis=1) # [M, M']

    def cross_covariance(self, xi, t, theta):

        r"""
        Computes K_{Fy}\left( \xi, t^{*}\right),
        which is the cross-covariance between the time-domain
        and the frequency domain.
        
        Corresponds to Real and Imaginary part of equation ... 
        """
        #NOTE -- this will only work for 1D data!
        _alpha = tf.reshape(self.kernel.alpha, [-1,]) # [1,]
        _gamma = tf.reshape(self.kernel.bandwidths, [-1,]) # [Q,]
        _sigma = tf.reshape(self.kernel.powers, [-1,]) # [Q,]
        _theta = tf.reshape(theta, [-1,]) # [Q,]

        _pi = np.pi

        at = _alpha / _pi**2 # [1,]
        gt = _gamma / _pi**2 # [Q,]
        L = 1./at + 1./gt # [Q,]

        # prepare for broadcasting
        at = at[tf.newaxis, :, tf.newaxis] # [1, 1, 1]
        gt = gt[tf.newaxis, :, tf.newaxis] # [1, Q, 1]
        L = L[tf.newaxis, :, tf.newaxis] # [1, Q, 1]

        """
        def time_freq_SM_re(x, y, alpha, gamma, theta, sigma=1):
            at = alpha/(np.pi**2)
            gt = gamma/(np.pi**2)
            L = 1/at + 1/gt
            (sigma**2)/(np.sqrt(np.pi*(at+gt))) * 
            np.exp(outersum(-(x-theta)**2/(at+gt), -y**2*np.pi**2/L) ) *
            np.cos(-np.outer(2*np.pi*(x/at+theta/gt)/(1/at + 1/gt),y))
        """

        Kfu = _sigma[tf.newaxis, :, tf.newaxis]**2 / tf.sqrt(_pi * (at+gt)) # [1,Q,1]

        Kfu*= tf.math.exp(- tf.square(tf.reshape(xi, [-1,1,1])
                                    - tf.reshape(_theta, [1,-1,1])
                                    ) # [M, Q, 1] 
                            / (at+gt) - tf.square(tf.reshape(t, [1, 1, -1])) # [N, 1, 1]
                        * _pi**2 / L) # [M, Q, 1]

        #NOTE -- why do we have a minus sign inside the cosine?
        Kfu_real = Kfu * tf.math.cos(- 2*_pi* (tf.reshape(xi, [-1,1,1]) / at # [M, Q, 1] 
                            + tf.reshape(_theta, [1,-1,1]) / gt # [1, Q, 1]
                            )
                            / L # [M, Q, 1] 
                            * tf.reshape(t, [1,1,-1]) # [1, 1, N]
                            ) # [M, Q, N]

        Kfu_img = Kfu * tf.math.sin(- 2*_pi* (tf.reshape(xi, [-1,1,1]) / at # [M, Q, 1] 
                            + tf.reshape(_theta, [1,-1,1]) / gt # [1, Q, 1]
                            )
                            / L # [M, Q, 1] 
                            * tf.reshape(t, [1,1,-1]) # [1, 1, N]
                            ) # [M, Q, N]

        return tf.reduce_sum(Kfu_real, axis=1), tf.reduce_sum(Kfu_img, axis=1) # [M,N]


    #NOTE -- this is redundant as I can just use self.kernel(.,.)

    def Spec_Mix_tf(self, x, y):
        
        #Spectral Mixture Kernel.

        #Main reference is Wilson, 2018.

        #Corresponds to equation 15 in the BNSE paper.
    
        _pi = np.pi

        #Tobar version --only works for Q=1 it seems
        #exp_part = tf.exp(-_gamma * tf.square(outersum(x,-y)))
        #cosine_part = tf.cos(2. * _pi * _theta * outersum(x,-y))
        #return _sigma**2 * exp_part * cosine_part


        X_scaled = x[...,tf.newaxis] * self.kernel.means[tf.newaxis,...] # [N, D, Q]
        Y_scaled = y[...,tf.newaxis] * self.kernel.means[tf.newaxis,...] # [N, D, Q]

        X_scaled = tf.transpose(X_scaled, [2,0,1]) # [Q, N1, D]
        Y_scaled = tf.transpose(Y_scaled, [2,0,1]) # [Q, N2, D]

        d = X_scaled[:,:,tf.newaxis,:]  - Y_scaled[:,tf.newaxis,:,:] # [Q, N1, N2, D]

        cos_term = tf.cos(2 * _pi * tf.reduce_sum(d, axis = -1)) # expected shape -- [Q, N1, N2]
        
        # means - [D, Q]       
        descaled_d = d * tf.transpose(tf.math.reciprocal(self.kernel.means)
                                      )[:, tf.newaxis, tf.newaxis, :] # expected shape [Q, N1, N2, D]
        
        # bandwidths - [D, Q]
        exponential_term = tf.exp(- 
                                  tf.reduce_sum(tf.square(descaled_d) * 
                                                tf.transpose(self.kernel.bandwidths)[:, tf.newaxis, tf.newaxis, :]
                                                , axis=-1)) # expected shape [Q, N1, N2]


        return self.kernel.powers**2 * tf.reduce_sum(exponential_term, axis = 0) * tf.reduce_sum(cos_term, axis = 0) 

        
    def time_freq_covariances(self, xi, t, kernel = 'sm'):

        #NOTE -- this will only work for 1D data!

        if kernel == 'sm':

            pos_freq_real, pos_freq_img = self.cross_covariance(xi, t, self.kernel.means)
            neg_freq_real, neg_freq_img = self.cross_covariance(xi, t, -self.kernel.means)

            tf_real_cov = 0.5*(pos_freq_real + neg_freq_real)
            tf_imag_cov = 0.5*(pos_freq_img + neg_freq_img)

        return tf_real_cov, tf_imag_cov

    def set_labels(self, time_label, signal_label):
        self.time_label = time_label
        self.signal_label = signal_label

    def set_freqspace(self, max_freq, dimension=500):
        self.w = np.linspace(0, max_freq, dimension)

#TODO -- maybe move this to a maths util 
def outersum(a, b):
    _ = tf.experimental.numpy.outer(a, tf.ones_like(b))
    __ = tf.experimental.numpy.outer(tf.ones_like(a), b)

    return _ + __



def outersum_np(a,b):
    return np.outer(a,np.ones_like(b))+np.outer(np.ones_like(a),b)