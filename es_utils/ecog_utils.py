import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def gridOrient(subject):

    if subject in ['EC183','EC194','EC195','EC172','EC36']:
        gridOrient_zeroIndexed = np.rot90(np.reshape(range(256),(16,16)),2).T
    elif subject in ['EC186','EC188','EC199','EC142']:
        gridOrient_zeroIndexed = np.vstack([np.flipud(np.reshape(range(128),(8,16))),np.flipud(np.reshape(range(128,256),(8,16)))])
    elif subject in ['EC193','EC208']:
        gridOrient_zeroIndexed = np.rot90(np.vstack([np.flipud(np.reshape(range(128),(8,16))),np.flipud(np.reshape(range(128,256),(8,16)))]))
    else:
        raise RuntimeError(f'gridOrient not defined for subject {subject}')

    return gridOrient_zeroIndexed


def compute_time_in_trial(T,Fs,stim_onset_inds,max_trial_len,trial_onset):
    assert all([s2>s1 for s1,s2 in zip(stim_onset_inds[:-1],stim_onset_inds[1:])]),\
           'stim_onset_inds must be sorted.'
    trial_window = np.arange(Fs*max_trial_len)+np.ceil(Fs*trial_onset)
    trial_timeaxis = trial_window / Fs

    time_in_trial = np.zeros(T)*np.nan
    trial_number = np.zeros(T)*np.nan
    for i,stimOnset in enumerate(stim_onset_inds):
        dat_inds = (stimOnset + trial_window).astype(int)
        time_in_trial[dat_inds] = trial_timeaxis # may overwrite the end of the previous trial
        trial_number[dat_inds] = i

    return time_in_trial,trial_number


def imshow_on_electrode_grid(toplot,gridOrient_0indexed,ylabels=None,titles=None,fignum=None,cl=None,cmap='RdBu_r',
                             colorbar='horizontal',blnElectrodeNumbers=False):
    # There will be nrows x ncols subplots
    # Each subplot will have an imshow with the electrodes laid out according to gridOrient_0indexed
    # toplot: nrows x ncols x N
    nrows,ncols,N = toplot.shape

    if ylabels is None:
        ylabels = ['']*nrows
    if titles is None:
        titles = ['']*ncols

    assert(len(titles)==ncols)
    assert(len(ylabels)==nrows)

    f=plt.figure(fignum)
    f.clf()
    if cl is None:
        cl = np.array([-1,1])*np.nanmax(np.abs(toplot))
    if colorbar=='horizontal':
        gs = gridspec.GridSpec(nrows+1,ncols, height_ratios=[1]*nrows+[0.2])
    elif colorbar=='vertical':
        gs = gridspec.GridSpec(nrows,ncols+2, width_ratios=[1]*ncols+[0.2,0.8])
    else:
        gs = gridspec.GridSpec(nrows,ncols)

    for i,ylabel in enumerate(ylabels):
        for j,title in enumerate(titles):
            ax = f.add_subplot(gs[i,j])
            h=ax.imshow(toplot[i,j,gridOrient_0indexed],interpolation='nearest',vmin=cl[0],vmax=cl[1],cmap=cmap)
            if i==0:
                ax.set_title(title)
            if j==0:
                ax.set_ylabel(ylabel)
            if blnElectrodeNumbers:
                for k in range(gridOrient_0indexed.shape[0]):
                    for l in range(gridOrient_0indexed.shape[1]):
                        ax.text(l,k,int(gridOrient_0indexed[k,l])+1,
                                horizontalalignment='center',verticalalignment='center',fontsize=6)

    if colorbar=='horizontal':
        ax = f.add_subplot(gs[nrows,:])
        cb = f.colorbar(h,cax=ax,orientation='horizontal')
    elif colorbar=='vertical':
        ax = f.add_subplot(gs[:,ncols])
        cb = f.colorbar(h,cax=ax,orientation='vertical')
    else:
        cb = None

    return f,cb

def electrode_grid_subplots(toplot,gridOrient_0indexed,plotting_function,fignum=None,
                            blnAxesOff=True,blnTitleNumber=True):
    # There will be one subplot per electrode
    # Each subplot will call the plotting_function for the data in toplot[i] with i the electrode index
    # toplot: N x ?? (electrodes in first dimension, other(s) can be anything)

    nrows,ncols = gridOrient_0indexed.shape
    gs = gridspec.GridSpec(nrows, ncols)

    f = plt.figure(fignum,[14,8])
    f.clf()
    for i in range(nrows):
        for j in range(ncols):
            index = gridOrient_0indexed[i,j]
            ax = f.add_subplot(gs[i, j])
            plotting_function(toplot[index])
            if blnAxesOff:
                ax.axis('off')
            if blnTitleNumber:
                ax.text(0.5, 0.5,index+1,
                    horizontalalignment='center',verticalalignment='center',
                    transform=ax.transAxes)

    return f
