
#NOTE -- this is from Tobar's package
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import find_peaks, peak_widths
from gpflow.models import BNSE
from gpflow.kernels import MixtureSpectralGaussian
import statsmodels.api as sm
import gpflow
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
#EXPERIMENT = 'hr1'
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

time = time.reshape((-1,1))
time = time - np.median(time)
signal = signal.reshape((-1,1))

MODEL = 'GPR'
KERNEL = 'SpecMixGaus'
N_FREQS = 500
model = BNSE(data = (time, signal))
model.set_labels(time_label, signal_label)

if EXPERIMENT=='hr1':
    model.set_freqspace(0.03, N_FREQS)

elif EXPERIMENT == 'hr2':
    model.set_freqspace(0.03, N_FREQS)

elif EXPERIMENT == 'sunspots':
    model.set_freqspace(0.2, N_FREQS)

# Training 
optimise = True
MAXITER=100
if optimise:
    #gpflow.set_trainable(model.likelihood, False)

    opt_logs = gpflow.optimizers.Scipy().minimize(
        model.training_loss,
        variables=model.trainable_variables,
        method="l-bfgs-b",
        options={"disp": True, "maxiter": MAXITER},
    )

noise_variance = model.likelihood.variance.numpy()
print(f'Negative log likelihood (before training): {noise_variance}')

#nll = model.neg_log_likelihood()
#print(f'Negative log likelihood (before training): {nll}')
_time = np.linspace(np.min(time), np.max(time), 500)
_time = _time.reshape((-1,1))
_w = np.linspace(0, N_OBSERVATIONS / (np.max(time) - np.min(time)) / 16., 500)
model.compute_moments(Xnew=_time)

def plot_time_posterior(X, 
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
    plt.plot(Xnew.ravel(), model.post_mean.numpy().ravel(), color='blue', label='posterior mean')
    
    error_bars = 2 * np.sqrt(np.diag(model.post_cov.numpy())).ravel()
    plt.fill_between(Xnew.ravel(), model.post_mean.numpy().ravel() - error_bars, model.post_mean.numpy().ravel() + error_bars, 
                        color='blue', alpha=0.1, label='95% error bars')
    if flag == 'with_window':
        plt.plot(Xnew.ravel(), 2.*model.kernel.powers * np.exp(-model.kernel.alpha * Xnew**2))
    
    plt.title('Observations and posterior interpolation')
    plt.xlabel(time_label)
    plt.ylabel(signal_label)
    plt.legend()
    plt.xlim([min(X),max(X)])
    plt.tight_layout()
    plt.savefig(f'./figures/{MODEL}_{KERNEL}_time_domain_posterior_{EXPERIMENT}.png')
    plt.close()

plot_time_posterior(X = time,
                    Y = signal,
                    Xnew = _time,
                    model = model,
                    time_label=time_label,
                    signal_label=signal_label,
                    flag='with_window'
                    )
#plt.savefig("posterior_time.pdf", bbox_inches='tight', pad_inches=0)


def plot_freq_posterior_real(model):
    plt.figure(figsize=(18,6))
    plt.plot(model.w.ravel(), model.post_mean_r.numpy().ravel(), color='blue', label='posterior mean')
    

    print('---- post_cov_r ----')
    print(model.post_cov_r.numpy())

    error_bars = 2 * np.sqrt((np.diag(model.post_cov_r.numpy())))
    plt.fill_between(model.w.ravel(), model.post_mean_r.numpy().ravel() - error_bars.ravel(), 
                     model.post_mean_r.numpy().ravel() + error_bars.ravel(), color='blue',
                     alpha=0.1, label='95% error bars')
    
    plt.title('Posterior spectrum (real part)')
    plt.xlabel('frequency')
    plt.legend()
    plt.xlim([min(model.w.ravel()), max(model.w.ravel())])
    plt.tight_layout()
    plt.savefig(f'./figures/{MODEL}_{KERNEL}_freq_domain_posterior_real_{EXPERIMENT}.png')
    plt.close()


def plot_freq_posterior_imag(model):
    plt.figure(figsize=(18,6))
    plt.plot(model.w.ravel(), model.post_mean_i.numpy().ravel(), color='blue', label='posterior mean')
    
    print('---- post_cov_i ----')
    print(model.post_cov_i.numpy())

    error_bars = 2. * np.sqrt((np.diag(model.post_cov_i.numpy())))
    plt.fill_between(model.w.ravel(), model.post_mean_i.numpy().ravel() - error_bars.ravel(), 
                     model.post_mean_i.numpy().ravel() + error_bars.ravel(), color='blue',
                     alpha=0.1, label='95% error bars')
    
    plt.title('Posterior spectrum (imaginary part)')
    plt.xlabel('frequency')
    plt.legend()
    plt.xlim([min(model.w.ravel()),max(model.w.ravel())])
    plt.tight_layout()
    plt.savefig(f'./figures/{MODEL}_{KERNEL}_freq_domain_posterior_img_{EXPERIMENT}.png')
    plt.close()



plot_freq_posterior_real(model=model)
#plt.savefig("posterior_spectrum_real.pdf", bbox_inches='tight', pad_inches=0)
plot_freq_posterior_imag(model=model)
#plt.savefig("posterior_spectrum_imag.pdf", bbox_inches='tight', pad_inches=0)


#TODO -- need to fix this at one point
def plot_power_spectral_density(model, how_many, flag=None):
    #posterior moments for frequency
    plt.figure(figsize=(18,6))
    _w = model.w
    freqs = len(_w)
    samples = np.zeros((freqs,how_many))
    
    #convert to numpy
    _post_mean_r = model.post_mean_r.numpy().ravel()
    _post_mean_i = model.post_mean_i.numpy().ravel()
    _post_cov_r = model.post_cov_r.numpy()
    _post_cov_i = model.post_cov_i.numpy()
    _post_mean_F = model.post_mean_F.numpy().ravel()
    _post_cov_F = model.post_cov_F.numpy()
    
    for i in range(how_many):               
        sample = np.random.multivariate_normal(_post_mean_F, (_post_cov_F + _post_cov_F.T)/2. 
                                               + 1e-5*np.eye(2*freqs))
        samples[:,i] = sample[0:freqs]**2 + sample[freqs:]**2
    plt.plot(_w, samples, color='red', alpha=0.35)
    plt.plot(_w, samples[:,0], color='red', alpha=0.35, label='posterior samples')
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
    plt.savefig(f'./figures/{MODEL}_{KERNEL}_psd_{EXPERIMENT}.png')
    plt.close()
    if flag == 'show peaks':
        return peaks, widths

peaks, widths = plot_power_spectral_density(model, 15, 'show peaks')
#plt.savefig("posterior_psd.pdf", bbox_inches='tight', pad_inches=0)

print(f'Peaks are at positions {peaks*(model.w[1]-model.w[0]) }')
print(f'and their widths are {widths[0]*(model.w[1]-model.w[0])}')


def plot_complex_gp_covariances(model):
    """
    #TODO -- write documentation
    """
    _pcov, _cov = model.get_complex_gp_covariances()

    print(_pcov)
    print(_cov)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 10))

    # Pseudo-covariance local spectrum \mathcal{F}_{c}(xi,xi') -- Complex GP
    ax1.matshow(_pcov.numpy(), aspect="auto")
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.set_title("Pseudo-covariance")

    # Covariance local spectrum \mathcal{F}_{c}(xi,xi') -- Complex GP
    ax2.matshow(_cov.numpy(), aspect="auto")
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.set_title("Covariance")

    plt.savefig(f'./figures/{MODEL}_{KERNEL}_{EXPERIMENT}_complex_gp_covariances.png')
    plt.close()

plot_complex_gp_covariances(model)
