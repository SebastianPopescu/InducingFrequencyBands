
#NOTE -- this is from Tobar's package
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.signal import find_peaks, peak_widths
from gpflow.models import BNSE
from gpflow.kernels import MixtureSpectralGaussian
#import scipy.signal as sp
#sns.set_style("whitegrid")
#import scipy.signal as sp
#import spectrum as spectrum

plot_params = {'legend.fontsize': 18,
          'figure.figsize': (15, 5),
         'xtick.labelsize':'18',
         'ytick.labelsize':'18',
         'axes.titlesize':'24',
         'axes.labelsize':'22'}
plt.rcParams.update(plot_params)


EXPERIMENT = 'hr1'
#EXPERIMENT = 'hr2'
#EXPERIMENT = 'sunspots'

if EXPERIMENT=='hr1':
    signal = np.loadtxt('./docs/notebooks/data/hr2.txt') 
    time = (np.linspace(0, 1800,1800))
    time_label = 'time'
    signal_label = 'heart-rate signal'

elif EXPERIMENT == 'hr2':
    signal = np.loadtxt('./docs/notebooks/data/hr1.txt') 
    time = (np.linspace(0, 1800,1800))
    time_label = 'time'
    signal_label = 'heart-rate signal'

elif EXPERIMENT == 'sunspots':
    dta = sm.datasets.sunspots.load_pandas().data
    signal = np.array(dta.SUNACTIVITY)
    time = np.array(dta.YEAR)
    time_label = 'time'
    signal_label = 'sunspot data'
        
signal = signal - np.mean(signal)
#you can change the number of observations here
#indices = np.random.randint(0, len(time), size=int(len(time))) 
N_OBSERVATIONS = 600
indices = np.random.randint(0, len(time), size=N_OBSERVATIONS) 
indices = np.sort(indices)
signal = signal[indices]
time = time[indices]

MODEL = 'GPR'


kern = MixtureSpectralGaussian(n_components=,
                               means= ,
                               bandwidths=,
                               powers=)
my_bse = BNSE(data = (time, signal))
my_bse.set_labels(time_label, signal_label)

if EXPERIMENT=='hr1':
    my_bse.set_freqspace(0.03)

elif EXPERIMENT == 'hr2':
    my_bse.set_freqspace(0.03)

elif EXPERIMENT == 'sunspots':
    my_bse.set_freqspace(0.2)




my_bse.train()

#after training

nll = my_bse.neg_log_likelihood()
print(f'Negative log likelihood (before training): {nll}')
my_bse.compute_moments()
my_bse.plot_time_posterior()
#plt.savefig("posterior_time.pdf", bbox_inches='tight', pad_inches=0)


"""
my_bse.plot_freq_posterior_real()
plt.savefig("posterior_spectrum_real.pdf", bbox_inches='tight', pad_inches=0)
my_bse.plot_freq_posterior_imag()
plt.savefig("posterior_spectrum_imag.pdf", bbox_inches='tight', pad_inches=0)

my_bse.plot_power_spectral_density(15)

peaks, widths = my_bse.plot_power_spectral_density(15, 'show peaks');
plt.savefig("posterior_psd.pdf", bbox_inches='tight', pad_inches=0)

print(f'Peaks are at positions {peaks*(my_bse.w[1]-my_bse.w[0]) }')
print(f'and their widths are {widths[0]*(my_bse.w[1]-my_bse.w[0])}')
"""


##################################
##################################
### Plotting functions ###########
##################################
##################################

def plot_time_posterior(self, 
                        X, 
                        Y, 
                        Xnew, 
                        model, 
                        time_label,
                        signal_label,
                        flag=None):

    """
    Posterior moments for time-domain.
    Boils down to standard GP conditional plots.

    Options for flag : 'with_window'
    """
    
    #posterior moments for time domain 
    plt.figure(figsize = (18, 6))
    plt.plot(X, Y,'.r', markersize=10, label='observations')
    plt.plot(Xnew, model.post_mean, color='blue', label='posterior mean')
    
    error_bars = 2 * np.sqrt(np.diag(model.post_cov))
    plt.fill_between(Xnew, model.post_mean - error_bars, model.post_mean + error_bars, 
                        color='blue', alpha=0.1, label='95% error bars')
    if flag == 'with_window':
        plt.plot(Xnew, 2.*model.kernel.sigma * np.exp(-model.kernel.alpha * Xnew**2))
    
    plt.title('Observations and posterior interpolation')
    plt.xlabel(time_label)
    plt.ylabel(signal_label)
    plt.legend()
    plt.xlim([min(self.x),max(self.x)])
    plt.tight_layout()
    plt.savefig(f'./figures/{MODEL}_{KERNEL}_{INIT_METHOD}_time_domain_posterior.png')
    plt.close()


def plot_freq_posterior_real(model):
    plt.figure(figsize=(18,6))
    plt.plot(model.w, model.post_mean_r, color='blue', label='posterior mean')
    
    error_bars = 2 * np.sqrt((np.diag(model.post_cov_r)))
    plt.fill_between(model.w, model.post_mean_r - error_bars, 
                     model.post_mean_r + error_bars, color='blue',
                     alpha=0.1, label='95% error bars')
    
    plt.title('Posterior spectrum (real part)')
    plt.xlabel('frequency')
    plt.legend()
    plt.xlim([min(model.w), max(model.w)])
    plt.tight_layout()
    plt.savefig(f'./figures/{MODEL}_{KERNEL}_{INIT_METHOD}_freq_domain_posterior_real.png')
    plt.close()


def plot_freq_posterior_imag(model):
    plt.figure(figsize=(18,6))
    plt.plot(model.w, model.post_mean_i, color='blue', label='posterior mean')
    
    error_bars = 2. * np.sqrt((np.diag(model.post_cov_i)))
    plt.fill_between(model.w, model.post_mean_i - error_bars, 
                     model.post_mean_i + error_bars, color='blue',
                     alpha=0.1, label='95% error bars')
    
    plt.title('Posterior spectrum (imaginary part)')
    plt.xlabel('frequency')
    plt.legend()
    plt.xlim([min(model.w),max(model.w)])
    plt.tight_layout()
    plt.savefig(f'./figures/{MODEL}_{KERNEL}_{INIT_METHOD}_freq_domain_posterior_img.png')
    plt.close()

def plot_freq_posterior():
    plot_freq_posterior_real()
    plot_freq_posterior_imag()
 
def plot_power_spectral_density_old(model, how_many, flag=None):
    
    #posterior moments for frequency
    plt.figure(figsize=(18,6))
    _w = model.w.numpy()
    freqs = len(_w)
    samples = np.zeros((freqs,how_many))

    #convert to numpy
    _post_mean_r = model.post_mean_r.numpy()
    _post_mean_i = model.post_mean_i.numpy()
    _post_cov_r = model.post_cov_r.numpy()
    _post_cov_i = model.post_cov_i.numpy()

    for i in range(how_many):               
        sample_r = np.random.multivariate_normal(_post_mean_r, (_post_cov_r+_post_cov_r.T)/2 + 1e-5*np.eye(freqs))
        sample_i = np.random.multivariate_normal(_post_mean_i, (_post_cov_i+_post_cov_i.T)/2 + 1e-5*np.eye(freqs))
        samples[:,i] = sample_r**2 + sample_i**2
    plt.plot(_w, samples, color='red', alpha=0.35)
    plt.plot(_w, samples[:,0], color='red', alpha=0.35, label='posterior samples')
    
    posterior_mean_psd = _post_mean_r**2 + _post_mean_i**2 + np.diag(_post_cov_r + _post_cov_i)
    plt.plot(_w,posterior_mean_psd, color='black', label = '(analytical) posterior mean')
    if flag == 'show peaks':
        peaks, _  = find_peaks(posterior_mean_psd, prominence=500000)
        widths = peak_widths(posterior_mean_psd, peaks, rel_height=0.5)
        plt.stem(_w[peaks],posterior_mean_psd[peaks], markerfmt='ko', label='peaks')
    plt.title('Sample posterior power spectral density')
    plt.xlabel('frequency')
    plt.legend()
    plt.xlim([min(self.w),max(self.w)])
    plt.tight_layout()
    plt.savefig(f'./figures/{MODEL}_{KERNEL}_{INIT_METHOD}_psd_old_version.png')
    plt.close()
    if flag == 'show peaks':
        return peaks, widths
        

def plot_power_spectral_density(model, how_many, flag=None):
    #posterior moments for frequency
    plt.figure(figsize=(18,6))
    _w = model.w.numpy()
    freqs = len(_w)
    samples = np.zeros((freqs,how_many))
    
    #convert to numpy
    _post_mean_r = model.post_mean_r.numpy()
    _post_mean_i = model.post_mean_i.numpy()
    _post_cov_r = model.post_cov_r.numpy()
    _post_cov_i = model.post_cov_i.numpy()
    _post_mean_F = model.post_mean_F.numpy()
    _post_cov_F = model.post_cov_F.numpy()
    
    for i in range(how_many):               
        sample = np.random.multivariate_normal(_post_mean_F,(_post_cov_F + _post_cov_F.T)/2 
                                               + 1e-5*np.eye(2*freqs))
        samples[:,i] = sample[0:freqs]**2 + sample[freqs:]**2
    plt.plot(_w,samples, color='red', alpha=0.35)
    plt.plot(_w,samples[:,0], color='red', alpha=0.35, label='posterior samples')
    posterior_mean_psd = _post_mean_r**2 + _post_mean_i**2 + np.diag(_post_cov_r + _post_cov_i)
    plt.plot(_w, posterior_mean_psd, color='black', label = '(analytical) posterior mean')
    if flag == 'show peaks':
        peaks, _  = find_peaks(posterior_mean_psd, prominence=500000)
        widths = peak_widths(posterior_mean_psd, peaks, rel_height=0.5)
        plt.stem(_w[peaks],posterior_mean_psd[peaks], markerfmt='ko', label='peaks')
    
    plt.title('Sample posterior power spectral density')
    plt.xlabel('frequency')
    plt.legend()
    plt.xlim([min(_w), max(_w)])
    plt.tight_layout()
    plt.savefig(f'./figures/{MODEL}_{KERNEL}_{INIT_METHOD}_psd.png')
    plt.close()
    if flag == 'show peaks':
        return peaks, widths


def set_freqspace(max_freq, dimension=500):
    _w = np.linspace(0, max_freq, dimension)
    return _w