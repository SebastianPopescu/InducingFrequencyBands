import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def mean_absolute_error(y_true, y_pred):
    """
    Mean Absolute Error (MAE) between the true and the predicted values.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE) between the true and the predicted values.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    idx = 1e-6 < y_true
    y_true, y_pred = y_true[idx], y_pred[idx]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) between the true and the predicted values.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    idx = 1e-6 < y_true
    y_true, y_pred = y_true[idx], y_pred[idx]
    return np.mean(np.abs((y_true - y_pred) / (y_true + y_pred))) * 200.0

def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error (MSE) between the true and the predicted values.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred)**2)

def root_mean_squared_error(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE) between the true and the predicted values.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred)**2))

def plot_spectrum(means, scales, dataset=None, weights=None, noises=None, method='LS', maxfreq=None, log=False, n=10000, titles=None, show=True, filename=None, title=None):
    """
    Plot spectral Gaussians of given means, scales and weights.
    """
    if means.ndim == 2:
        means = np.expand_dims(means, axis=2)
    if scales.ndim == 2:
        scales = np.expand_dims(scales, axis=2)
    if isinstance(weights, np.ndarray) and weights.ndim == 1:
        weights = np.expand_dims(weights, axis=1)
    if isinstance(maxfreq, np.ndarray) and maxfreq.ndim == 1:
        maxfreq = np.expand_dims(maxfreq, axis=1)

    if means.ndim != 3:
        raise ValueError('means and scales must have shape (mixtures,output_dims,input_dims)')
    if means.shape != scales.shape:
        raise ValueError('means and scales must have the same shape (mixtures,output_dims,input_dims)')
    if noises is not None and (noises.ndim != 1 or noises.shape[0] != means.shape[1]):
        raise ValueError('noises must have shape (output_dims,)')
    if dataset is not None and len(dataset) != means.shape[1]:
        raise ValueError('means and scales must have %d output dimensions' % len(dataset))

    mixtures = means.shape[0]
    output_dims = means.shape[1]
    input_dims = means.shape[2]

    if isinstance(weights, np.ndarray) and (weights.ndim != 2 or weights.shape[0] != mixtures or weights.shape[1] != output_dims):
        raise ValueError('weights must have shape (mixtures,output_dims)')
    elif not isinstance(weights, np.ndarray):
        weights = np.ones((mixtures,output_dims))
    if isinstance(maxfreq, np.ndarray) and (maxfreq.ndim != 2 or maxfreq.shape[0] != output_dims or maxfreq.shape[1] != input_dims):
        raise ValueError('maxfreq must have shape (output_dims,input_dims)')


    h = 4.0*output_dims
    fig, axes = plt.subplots(output_dims, input_dims, figsize=(12,h), squeeze=False, constrained_layout=True)
    if title is not None:
        fig.suptitle(title, y=(h+0.8)/h, fontsize=18)
    
    for j in range(output_dims):
        for i in range(input_dims):
            x_low = max(0.0, norm.ppf(0.01, loc=means[:,j,i], scale=scales[:,j,i]).min())
            x_high = norm.ppf(0.99, loc=means[:,j,i], scale=scales[:,j,i]).max()

            if dataset is not None:
                maxf = maxfreq[j,i] if maxfreq is not None else None
                dataset[j].plot_spectrum(ax=axes[j,i], method=method, transformed=True, n=n, log=False, maxfreq=maxf)
                x_low = axes[j,i].get_xlim()[0]
                x_high =  axes[j,i].get_xlim()[1]
            if maxfreq is not None:
                x_high = maxfreq[j,i]

            psds = []
            x = np.linspace(x_low, x_high, n)
            psd_total = np.zeros(x.shape)
            for q in range(mixtures):
                psd = weights[q,j] * norm.pdf(x, loc=means[q,j,i], scale=scales[q,j,i])
                axes[j,i].axvline(means[q,j,i], ymin=0.001, ymax=0.05, lw=3, color='r')
                psd_total += psd
                psds.append(psd)
            if noises is not None:
                psd_total += noises[j]**2

            for psd in psds:
                psd /= psd_total.sum()*(x[1]-x[0]) # normalize
                axes[j,i].plot(x, psd, ls='--', c='b')
            psd_total /= psd_total.sum()*(x[1]-x[0]) # normalize
            axes[j,i].plot(x, psd_total, ls='-', c='b')

            y_low = 0.0
            if log:
                x_low = max(x_low, 1e-8)
                y_low = 1e-8
            _, y_high = axes[j,i].get_ylim()
            y_high = max(y_high, 1.05*psd_total.max())
           
            axes[j,i].set_xlim(x_low, x_high)
            axes[j,i].set_ylim(y_low, y_high)
            axes[j,i].set_yticks([])
            if titles is not None:
                axes[j,i].set_title(titles[j])
            if log:
                axes[j,i].set_xscale('log')
                axes[j,i].set_yscale('log')

    axes[output_dims-1,i].set_xlabel('Frequency')

    legends = []
    if dataset is not None:
        legends.append(plt.Line2D([0], [0], ls='-', color='k', label='Data (LombScargle)'))
    legends.append(plt.Line2D([0], [0], ls='-', color='b', label='Model'))
    legends.append(plt.Line2D([0], [0], ls='-', color='r', label='Peak location'))
    fig.legend(handles=legends)

    if filename is not None:
        plt.savefig(filename+'.pdf', dpi=300)
    if show:
        plt.show()
    return fig, axes
