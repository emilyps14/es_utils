# Author: Emily P. Stephen <emilyps14@gmail.com>

import warnings
import nitime.algorithms as tsa
import numpy as np
import matplotlib.pyplot as plt



def quick_mtspecgram(data, fs, times, movingwin=(1,1), NW=2, NFFT=None, f_targets=None):
    """
    Compute a multitaper spectrogram of data
    Parameters
    ----------
    data : ndarray, (n_signals, n_timepoints)
        An array whose rows are timeseries.
    fs : The sampling frequency
    times : ndarray, (n_timepoints,)
        The time axis for data
    movingwin : (window,winstep) The length of the moving window and step size in seconds (set them equal to have no overlap)
        Default is (1,1), nonoverlapping 1s windows
    NW : The time-half-bandwidth product
        Default is 2, which leads to a frequency resolution of 4Hz for 1s windows
    Returns
    -------
    S : ndarray, (n_signals, n_frequencies, n_windows)
        The spectrogram of data
    freqs : ndarray, (n_frequencies,)
        The frequency axis for the spectrogram
    ta : ndarray, (n_windows,)
        The time axis for the spectrgram
    ntapers : The number of tapers used for the analysis (corresponding to
        > 90% energy concentration.)
    """
    Nv, Nt = data.shape

    Nwin,Nstep,inds = _get_sliding_windows(fs,movingwin,Nt)

    if NFFT is None:
        NFFT=Nwin

    freqs,findx = _process_freqaxis(fs, NFFT, f_targets, sides='onesided')
    Nf = freqs.shape[0]

    S = np.zeros((Nv, Nf, len(inds)))
    ta = np.zeros(len(inds))
    for i, starti in enumerate(inds):
        endi = starti + Nwin

        datachunk = data[:, starti:endi]

        datachunk_minusDC = np.subtract(datachunk, datachunk.mean(1,keepdims=True))

        _,psd,nu = tsa.multi_taper_psd(datachunk_minusDC,fs,NW=NW,sides='onesided',
                                       adaptive=False,jackknife=False,low_bias=True,NFFT=NFFT)

        S[:, :, i] = psd[:,findx]
        ta[i] = (times[starti] + Nwin/fs/2.)

    ntapers = nu[0,0]/2

    return S, freqs, ta, ntapers


def compute_windowed_fft(data,fs,times,movingwin,NW,K=None,NFFT=None, f_targets=None):
    """Compute a windowed fft
    Unlike chronux, does not divide by the sampling frequency
    Parameters
    ----------
    data : ndarray, (n_signals, n_timepoints)
        An array whose rows are timeseries.
    fs : The sampling frequency
    times : ndarray, (n_timepoints,)
        The time axis for data
    movingwin : (window,winstep) The length of the moving window and step size in seconds (set them equal to have no overlap)
        Default is (1,1), nonoverlapping 1s windows
    NW : The time-half-bandwidth product
        Default is 2, which leads to a frequency resolution of 4Hz for 1s windows
    K : The number of tapers to use
        Default is None, which uses all tapers with eigenvalues > 0.9
    NFFT : The number of FFT bins to compute (default: the number of samples in each window)
    Returns
    -------
    win_fft : ndarray, (n_signals, n_tapers, n_frequencies, n_windows)
        The windowed fft
    freqs : ndarray, (n_frequencies,)
        The frequency axis for the windowed fft
    ta : ndarray, (n_windows,)
        The time axis for the windowed fft
    eigvals : The eigenvalues of tapers used for the analysis
    """
    Nv, Nt = data.shape

    Nwin,Nstep,inds = _get_sliding_windows(fs,movingwin,Nt)

    if NFFT is None:
        NFFT=Nwin

    freqs,findx = _process_freqaxis(fs, NFFT, f_targets,sides='twosided')
    Nf = freqs.shape[0]

    # Precompute tapers
    if K is None:
        K = 2*NW
        low_bias = True
    else:
        low_bias = False

    args = (Nwin,NW,K)
    dpss, eigvals = tsa.dpss_windows(*args)
    if low_bias:
        keepers = (eigvals > 0.9)
        dpss = dpss[keepers]
        eigvals = eigvals[keepers]
    tapers = dpss
    K = tapers.shape[0]

    # Compute windowed FFT
    win_fft = np.zeros((Nv, K, Nf, len(inds)),dtype=np.complex)
    ta = np.zeros(len(inds))
    for i, starti in enumerate(inds):
        endi = starti + Nwin

        datachunk = data[:, starti:endi]

        datachunk_minusDC = np.subtract(datachunk, datachunk.mean(1)[:, np.newaxis])

        t_spectra = tsa.tapered_spectra(datachunk_minusDC,tapers,NFFT=NFFT,low_bias=False)

        win_fft[:, :, :, i] = t_spectra[:,:,findx]
        ta[i] = (times[starti] + Nwin/fs/2.) # 7/20/17 decided to add half window based on Nwin

    return win_fft, freqs, ta, eigvals


def _get_sliding_windows(fs,movingwin,Nt):
    Nwin = int(np.round(fs*movingwin[0])) # number of samples in window
    Nstep = np.round(fs*movingwin[1]) # number of samples to step through

    start_inds = list(range(0, int(Nt-Nwin+1), int(Nstep)))
    if start_inds[-1] + Nwin > Nt:
        start_inds = start_inds[0:-1]

    return Nwin,Nstep,start_inds


def _process_freqaxis(fs, NFFT, f_targets, sides):
    freqs = np.linspace(0, fs, NFFT, endpoint=False)
    if sides == 'onesided':
        # freqs = np.linspace(0, fs / 2, NFFT / 2 + 1)
        freqs = freqs[:int(np.floor(NFFT/2+1))]

    if f_targets is not None:
        findx = [np.argmin(np.abs(freqs-ftarg)) for ftarg in f_targets]
        freqs = freqs[findx]
        if len(f_targets)>1:
            tol = (f_targets[1]-f_targets[0])/100
        else:
            tol = f_targets[0]/100
        if not np.allclose(freqs,f_targets,atol=tol):
            warnings.warn('Frequency targets not found:\n f_targets: ' + str(f_targets) + '\n freqs:     ' + str(freqs),RuntimeWarning)
    else:
        findx = list(range(len(freqs)))

    return freqs,findx



def plot_specgram(S, ta, fa):
    plt.imshow(10 * np.log10(S.mean(0).squeeze()), aspect='auto', interpolation='none', origin='lower',
               extent=np.concatenate((ta[[0, -1]], fa[[0, -1]])))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()