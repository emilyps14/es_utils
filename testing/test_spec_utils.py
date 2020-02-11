import es_utils.spec_utils as utils
import numpy as np
import pytest

@pytest.mark.parametrize(['freqs','movingwin'],[([5, 10, 15, 20],(1,1)),
                                                ([8],(1,0.25))])
def test_quick_mtspecgram(freqs,movingwin):
    fs = 200
    T = 20
    window = movingwin[0]
    step = movingwin[1]
    NW = 2
    times = np.arange(0, T, 1. / fs)
    Nt = times.shape[0]
    Nf = len(freqs)

    data = np.zeros((Nf, Nt))
    for i, f in enumerate(freqs):
        data[i, :] = np.sin(2 * np.pi * f * times)

    S, fa, ta, ntapers = utils.quick_mtspecgram(data, fs, times, movingwin=movingwin, NW=NW)

    assert(ntapers == 3)
    assert(S.shape == (len(freqs), window * fs / 2 + 1, len(ta)))
    if window==step:
        assert(len(ta) == T / step)
    for i1, f1 in enumerate(freqs):
        Sf = S[i1, :, :]
        dataf_energy = np.sum(np.square(data[i1, :]))/fs
        finds = np.logical_and(fa>=f1-NW/window,fa<=f1+NW/window)

        if window==step:
            # Parseval's Theorem
            assert(np.isclose(np.sum(Sf), dataf_energy, rtol=1e-2))
            # Energy should be concentrated in the main lobe
            assert(np.isclose(np.sum(Sf[finds,:]), dataf_energy, rtol=1e-2))

        assert(np.allclose(Sf[np.logical_not(finds), :], 0, atol=1e-2))

@pytest.mark.parametrize(['freqs','movingwin','f_targets'],[([5, 10, 15, 20],(1,1),None),
                                                            ([8],(1,0.25),None),
                                                            ([15],(2,2),list(range(11))),
                                                            ([15],(2,2),[15])])
def test_compute_windowed_fft(freqs,movingwin,f_targets):
    fs = 200
    T = 20
    window = movingwin[0]
    step = movingwin[1]
    NW = 2
    times = np.arange(0, T, 1. / fs)
    Nt = times.shape[0]
    Nf = len(freqs)

    data = np.zeros((Nf, Nt))
    for i, f in enumerate(freqs):
        data[i, :] = np.sin(2 * np.pi * f * times)

    win_fft, fa, ta, eigvals = utils.compute_windowed_fft(data,fs,times,movingwin,NW,f_targets=f_targets)
    S, fa2, ta2, ntapers = utils.quick_mtspecgram(data, fs, times, movingwin=movingwin, NW=NW,f_targets=f_targets)

    assert(len(eigvals) == ntapers)
    if f_targets is None:
        assert(win_fft.shape == (len(freqs), ntapers, window * fs, len(ta)))
    else:
        assert(win_fft.shape == (len(freqs), ntapers, len(f_targets), len(ta)))
        assert(np.allclose(fa,f_targets))
    assert(all(np.equal(fa[:int(window*fs/2+1)],fa2)))
    assert(all(np.equal(ta,ta2)))
    for i1, f1 in enumerate(freqs):
        Sf = S[i1, :, :]
        fft_f = win_fft[i1, :, :int(window*fs/2+1), :]

        S2 = np.mean(fft_f*fft_f.conj(),axis=0)*2/fs
        assert np.allclose(Sf,S2,atol=1e-2)
