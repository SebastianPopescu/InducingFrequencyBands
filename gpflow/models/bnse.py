
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


from .. import posteriors
from ..base import InputData, MeanAndVariance, RegressionData, TensorData
from ..kernels import Kernel
from ..likelihoods import Gaussian
from ..logdensities import multivariate_normal
from ..mean_functions import MeanFunction
from ..utilities import add_likelihood_noise_cov, assert_params_false
from .model import TimeSpectrumBayesianModel
from .training_mixins import InternalDataTrainingLossMixin
from .util import data_input_to_tensor


class BNSE(TimeSpectrumBayesianModel, InternalDataTrainingLossMixin):

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
        if likelihood is None:
            if noise_variance is None:
                noise_variance = 1.0
            likelihood = Gaussian(noise_variance)
        _, Y_data = data

        self.initialise_kernel_hyperparams()
        kernel = self.set_kernel()
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1])
        self.data = data_input_to_tensor(data)

        if aim is None:
            """
            Aim==None, then...
            """
            X,Y = data            
            self.offset = np.median(X)
            self.x = X - self.offset
            self.y = Y
            #TODO -- do these need to me mentioned here?
            """
            self.post_mean = None
            self.post_cov = None
            self.post_mean_r = None
            self.post_cov_r = None
            self.post_mean_i = None
            self.post_cov_i = None
            self.time_label = None
            self.signal_label = None
            """


        elif aim == 'sampling':
            """
            Aim==None, then...
            """
            self.sigma = 1
            self.gamma = 1/2
            self.theta = 0
            self.sigma_n = 0

        elif aim == 'regression':
            """
            Aim==None, then...
            """
            X,Y = data
            self.x = X
            self.y = Y
            self.Nx = len(self.x)

    def initialise_kernel_hyperparams(self):
    
        self.Nx = len(self.x)
        self.alpha = 0.5 / ((np.max(self.x) - 
                             np.min(self.x))
                             /2.)**2
        self.sigma = np.std(self.y)
        self.gamma = 0.5 / ((np.max(self.x) - 
                             np.min(self.x))
                             /self.Nx)**2
        self.theta = 0.01
        self.sigma_n = np.std(self.y) / 10.
        self.time = np.linspace(np.min(self.x), np.max(self.x), 500)
        self.w = np.linspace(0, self.Nx / (np.max(self.x) - 
                                           np.min(self.x)) / 16., 500)


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
        
    #TODO -- adapt this to TF
    """
    def sample(self, X=None):

        if X is None: 
            self.Nx = 100
            self.x = np.random.random(self.Nx)
        elif np.size(X) == 1: 
            self.Nx = X
            self.x = np.random.random(self.Nx)
        else:
            self.x = X
            self.Nx = len(X)
        self.x = np.sort(self.x)
        cov_space = self.Spec_Mix(self.x, self.x, self.gamma, self.theta, self.sigma) + self.sigma_n**2 * tf.eye(self.Nx)

        #TODO -- need a TF or GPflow implementation 
        self.y =  np.random.multivariate_normal(np.zeros_like(self.x), cov_space)

        return self.y
    """
  
    """    
    #TODO -- maybe move to notebook
    #NOTE -- is this the auto-correlation function
    def acf(self, instruction):
        
        #TODO -- adapt this to TF and GPflow
        times = outersum(self.x, -self.x)
        corrs = np.outer(self.y, self.y)
        times = np.reshape(times, self.Nx**2)
        corrs = np.reshape(corrs, self.Nx**2)

        #aggregate for common lags
        t_unique = np.unique(times)
        #common_times = t_unique[:, np.newaxis] == times[:, np.newaxis].T
        common_times = np.isclose(t_unique[:, np.newaxis], times[:, np.newaxis].T)
        corrs_unique = np.dot(common_times,corrs)

        if instruction == 'plot.':
            plt.plot(t_unique, corrs_unique,'.')
        if instruction == 'plot-':
            plt.plot(t_unique, corrs_unique)

        return t_unique, corrs_unique
    """
        
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

        Kmm = self.Spec_Mix(X, X) + 1e-5*tf.eye(self.Nx) + self.sigma_n**2 * tf.eye(self.Nx)
        Knn = self.Spec_Mix(Xnew, Xnew)
        Kmn = self.Spec_Mix(X, Xnew)

        fmean, fvar = self.conditional_GP_equations(Kmm, Kmn, Knn, Y, full_cov)

        # original Tobar equations
        #self.post_mean = tf.squeeze(cov_star@np.linalg.solve(cov_space, self.y))
        #self.post_cov = cov_time - (cov_star@np.linalg.solve(cov_space, cov_star.T))
        return fmean, fvar

    def compute_moments(self, Xnew, X, Y, full_cov = False):

        #cov_space = Spec_Mix(self.x,self.x,self.gamma,self.theta,self.sigma) + 1e-5*np.eye(self.Nx) + self.sigma_n**2*np.eye(self.Nx)
        #cov_time = Spec_Mix(self.time,self.time, self.gamma, self.theta, self.sigma)
        #cov_star = Spec_Mix(self.time,self.x, self.gamma, self.theta, self.sigma)


        ### posterior moments for time domain ###
        post_mean, post_cov = self.compute_moments_Xnew(Xnew, X, Y, full_cov=full_cov)
    
        ### posterior moments for frequency domain ###
        cov_real, cov_imag = self.complex_gp_spectrum_covariances(self.w, self.w, kernel = 'sm')
        xcov_real, xcov_imag = self.time_freq_covariances(self.w, self.x, kernel = 'sm')
        
        #TODO -- this needs to be implemented in TF style
        Kmm = self.Spec_Mix(X, X) + 1e-5*tf.eye(self.Nx) + self.sigma_n**2 * tf.eye(self.Nx)

        #self.post_mean_r = np.squeeze(xcov_real@np.linalg.solve(cov_space, self.y))
        #self.post_cov_r = cov_real - (xcov_real@np.linalg.solve(cov_space, xcov_real.T))
        self.post_mean_r, self.post_cov_r = self.conditional_GP_equations(Kmm, 
                                                                          xcov_real, 
                                                                          cov_real, 
                                                                          Y, 
                                                                          full_cov=full_cov)

        self.post_mean_i, self.post_cov_i = self.conditional_GP_equations(Kmm, 
                                                                          xcov_imag, 
                                                                          cov_imag, 
                                                                          Y, 
                                                                          full_cov=full_cov)
        #TODO -- maybe this can be written somewhere else
        Lm = tf.linalg.cholesky(Kmm)
        A_right = tf.linalg.triangular_solve(Lm, xcov_imag, lower=True)  # [..., ?,?]
        A_left = tf.linalg.triangular_solve(Lm, xcov_real, lower=True)  # [..., ?,?]
        
        self.post_cov_ri = - tf.linalg.matmul(A_left, A_right, transpose_a=True)  # [..., ?,?]
        #post_cov_ri = - ((xcov_real@np.linalg.solve(Kmm, xcov_imag.T)))
        
        self.post_mean_F = tf.concat( [self.post_mean_r, self.post_mean_i], axis = -1) 
        self.post_cov_F = tf.experimental.numpy.vstack(
            (tf.experimental.numpy.hstack((self.post_cov_r, self.post_cov_ri)), 
             tf.experimental.numpy.hstack((self.post_cov_ri.T, self.post_cov_i))
             )) 
        
        return cov_real, xcov_real, Kmm
    
    def complex_gp_spectrum_covariances(self, x, y, kernel = 'sm'):
        
        if kernel == 'sm':
            N = len(x)
            # Local Spectrum \mathcal{F}_{c}\left(  \xi \right) 
            # Spectrum covariance
            K = 0.5*(self.spectrum_covariance(x, y) + 
                    self.spectrum_covariance(x, y))
            # Spectrum pseudo-covariance
            P = 0.5*(self.spectrum_covariance(x, -y) + 
                    self.spectrum_covariance(x, -y))
            # Krr -- real covariance
            real_cov = 0.5*(K + P) + 1e-8*tf.eye(N)
            # Kii -- imaginary covariance
            imag_cov = 0.5*(K - P) + 1e-8*tf.eye(N)
            #NOTE -- remainder: Kir = Kri = 0 since the underlying signal is real-valued.

        return real_cov, imag_cov

    def spectrum_covariance(self, xi1, xi2):

        """

        Computes K_{ff}\left( \xi, \xi' \right).

        Corresponds to equation 17 from BNSE paper.

        xi1 # [M,]
        xi2 # [M',]
        """
        _alpha = self.kernel.alpha # [Q,]
        _gamma = self.kernel.gamma # [Q,]
        _sigma = self.kernel.sigma # [Q,]
        _theta = self.kernel.theta # [Q,]

        _pi = tf.cast(np.pi, default_float())

        magnitude = _pi * _sigma**2 / (tf.sqrt(_alpha * (_alpha + 2.*_gamma))) # [Q,]
        magnitude = magnitude[tf.newaxis, :, tf.newaxis] # [1, Q, 1]
 
        Kxi_xi = tf.math.exp(-_pi**2/(2.*_alpha[tf.newaxis, :, tf.newaxis]) * #  [1, Q, 1]
                             tf.square(outersum(xi1,-xi2))[:,tf.newaxis,:] # [M, 1, M']
                             ) # [M, Q, M']
        Kxi_xi *= tf.math.exp( - 2.*_pi**2/
                              (_alpha + 2.*_gamma)[tf.newaxis, :, tf.newaxis] # [1, Q, 1]
                              * tf.square(outersum(xi1,xi2)[:,tf.newaxis,:]/2.
                                          -tf.reshape(_theta,[1,-1,1]))
                              ) # [M, Q, M']
        
        return tf.reduce_sum(magnitude * Kxi_xi, axis=1) # [M, M']


    def cross_covariance(self, xi, t, theta):

        """
        Computes K_{yF}\left( t^{*}, \xi \right),
        which is the cross-covariance between the time-domain
        and the frequency domain.
        
        Corresponds to Real and Imaginary part of equation ... 
        """

        _alpha = self.kernel.alpha # [Q,]
        _gamma = self.kernel.gamma # [Q,]
        _sigma = self.kernel.sigma # [Q,]

        _pi = tf.cast(np.pi, default_float())

        at = _alpha / _pi**2 # [Q,]
        gt = _gamma / _pi**2 # [Q,]
        L = 1./at + 1./gt # [Q,]

        # prepare for broadcasting
        at = at[tf.newaxis, :, tf.newaxis] # [1, Q, 1]
        gt = gt[tf.newaxis, :, tf.newaxis] # [1, Q, 1]
        L = L[tf.newaxis, :, tf.newaxis] # [1, Q, 1]

        Kfu = _sigma**2 / tf.sqrt(_pi * (at+gt)) # [Q,]
        # prepare for broadcasting
        Kfu = Kfu[tf.newaxis, :, tf.newaxis] # [1, Q, 1]

        #TODO -- need to be careful to use both negative and positive freqs in kernel.theta
        Kfu*= tf.math.exp(- tf.square(tf.reshape(xi, [1,1,-1])
                                    - tf.reshape(theta, [1,-1,1])
                                    ) # [1, Q, M] 
                            / (at+gt) ) # [1, Q, M]
        Kfu*= tf.math.exp(- tf.square(tf.reshape(t, [-1, 1, 1])) # [N, 1, 1]
                        * _pi**2 / L) # [N, Q, M]
        
        #NOTE -- why do we have a minus sign inside the cosine?
        Kfu_real = Kfu * tf.cos(- 2*_pi* (tf.reshape(xi, [1,1,-1]) / at # [1, Q, M] 
                            + tf.reshape(theta, [1,-1,1]) / gt # [1, Q, 1]
                            )
                            / L # [1, Q, M] 
                            * tf.reshape(t, [-1,1,1]) # [N, 1, 1]
                            ) # [N, Q, M]
        
        Kfu_img = Kfu * tf.sin(- 2*_pi* (tf.reshape(xi, [1,1,-1]) / at # [1, Q, M] 
                            + tf.reshape(theta, [1,-1,1]) / gt # [1, Q, 1]
                            )
                            / L # [1, Q, M] 
                            * tf.reshape(t, [-1,1,1]) # [N, 1, 1]
                            ) # [N, Q, M]

        return tf.reduce_sum(Kfu_real, axis=1), tf.reduce_sum(Kfu_img, axis=1) # [N, M]

    def Spec_Mix(self, x, y):
        
        """
        Spectral Mixture Kernel.

        Main reference is Wilson, 2018.

        Corresponds to equation 15 in the BNSE paper.
        """

        _pi = tf.cast(np.pi, default_float())

        exp_part = tf.exp(-self.kernel.gamma * 
                                             tf.square(outersum(x,-y)))

        cosine_part = tf.cos(2. * _pi * self.kernel.theta * outersum(x,-y))

        return self.kernel.sigma**2 * exp_part * cosine_part

    def Spec_Mix_sine(self, x, y, gamma, theta, sigma=1):
        
        """
        Spectral Mixture Kernel with a sine basis instead.
        #NOTE -- not sure what is the use for this.
        """

        return sigma**2 * np.exp(-gamma*outersum(x,-y)**2)*np.sin(2*np.pi*theta*outersum(x,-y))
        

    def time_freq_covariances(self, xi, t, kernel = 'sm'):
        if kernel == 'sm':

            pos_freq_real, pos_freq_img = self.cross_covariance(xi, t, self.kernel.theta)
            neg_freq_real, neg_freq_img = self.cross_covariance(xi, t, -self.kernel.theta)

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

