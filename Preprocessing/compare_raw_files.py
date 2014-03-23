#!/usr/bin/env python

import os, sys, mne
import pylab as pl
import numpy as np

stim_ch_name = ['STI 013']
event_id_stim, tmin, tmax = 1, -0.3, 0.3
event_id_ecg, event_id_eog = 999, 998
ecg_ch_name = 'ECG 001'
eog_ch_name = 'EOG 002'

# DISPLAY PARAMETERS
xlim1 = [-350, 350]
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

flow, fhigh = 1.0, 30
filter_type = 'butter'
filter_order = 4
njobs = 4

def rescale_artifact(evoked_meg_signal, evoked_artifact):
    b = (evoked_meg_signal.data.mean(axis=0).max() - evoked_meg_signal.data.mean(axis=0).min()) / (evoked_artifact.max() + evoked_artifact.min())
    a = evoked_meg_signal.data.mean(axis=0).max() * 1e15
    rescaled_artifact = evoked_artifact * b * 1e15 + a
    return rescaled_artifact # Only if it is a grad.

def calc_rms(data, average=None, rmsmean=None):
    #  check input
    sz = np.shape(data)
    nchan = np.size(sz)
    #  calc RMS
    rmsmean = 0
    if nchan == 1:
        ntsl = sz[0]
        rms = np.sqrt(np.sum(data**2)/ntsl)
        rmsmean = rms
    elif nchan == 2:
        ntsl = sz[1]
        powe = data**2
        rms = np.sqrt(sum(powe, 2)/ntsl)
        rmsmean = np.mean(rms)
        if average:
            rms = np.sqrt(np.sum(np.sum(powe, 1)/nchan)/ntsl)
    else: return -1
    return rms

def calc_performance(evoked_artifact, evoked_artifact_removed):
    diff_ecg = evoked_artifact.data - evoked_artifact_removed.data
    rms_diff_ecg = calc_rms(diff_ecg, average=1)
    rms_meg_ecg = calc_rms(evoked_artifact.data, average=1)
    arp_ecg = (rms_diff_ecg / rms_meg_ecg) * 100.0
    return arp_ecg

def difference_signal_plot(evoked_stim_before, evoked_stim_after, textstr_diff=None, save_title=None):
    stim_diff_ecg = evoked_stim_before - evoked_stim_after
    pl.figure('Difference signal with artifact rejection', figsize=(16, 10))
    a1 = pl.subplot(311)
    evoked_stim_before.plot(axes=a1)
    pl.title('Average epochs at stimulus onset')
    ylim_diff = dict(mag=a1.get_ylim())
    a2 = pl.subplot(312)
    evoked_stim_after.plot(axes=a2, ylim=ylim_diff)
    pl.title('Average epochs at stimulus onset with artifact rejection')
    a3 = pl.subplot(313)
    stim_diff_ecg.plot(axes=a3, ylim=ylim_diff)
    a3.text(0.05, 0.95, textstr_diff, transform=a3.transAxes, fontsize=14, 
            verticalalignment='top', bbox=props)
    pl.tight_layout()
    if save_title: pl.savefig(save_title, format='png')
    pl.close('Difference signal with artifact rejection')

def compare_raw_files(raw, raw_cleaned, raw_fname=None):
    pl.figure('Compare', figsize=(16, 10))
    
    # Plotting signal before ICA
    ecg_eve, _, _ = mne.preprocessing.find_ecg_events(raw, event_id_ecg, ch_name=ecg_ch_name)
    picks_meg = mne.fiff.pick_types(raw.info, meg=True, stim=False, exclude='bads')
    epochs = mne.Epochs(raw, ecg_eve, event_id_ecg, tmin, tmax, picks=picks_meg)
    evoked = epochs.average()
    
    ax1 = pl.subplot(221)
    evoked.plot(axes=ax1, xlim=xlim1)
    ylim_ecg = dict(mag=ax1.get_ylim())
    pl.title("Avg. MEG showing artifacts")
    textstr1 = 'num_events=%d\nEpochs: tmin, tmax = %0.1f, %0.1f\nRaw file name: %s' %(len(ecg_eve), tmin, tmax, raw)
    ax1.text(0.05, 0.95, textstr1, transform=ax1.transAxes, fontsize=12, 
             verticalalignment='top', bbox=props)
    
    # Plotting ecg signal
    picks_ecg = mne.fiff.pick_channels(raw.info['ch_names'], 'ECG 001')
    epochs_ecg = mne.Epochs(raw, ecg_eve, event_id_ecg, tmin, tmax, picks=picks_ecg)
    evoked_ecg = np.average(epochs_ecg.get_data(), axis=0).flatten()*-1
    pl.plot(evoked.times*1e3, rescale_artifact(evoked, evoked_ecg), 'r', axes=ax1)
    
    # Plotting signal after ICA
    epochs_after_ica = mne.Epochs(raw_cleaned, ecg_eve, event_id_ecg, tmin, tmax, picks=picks_meg)
    evoked_after_ica = epochs_after_ica.average()
    ax2 = pl.subplot(223)
    evoked_after_ica.plot(axes=ax2, ylim=ylim_ecg, xlim=xlim1)
    
    pl.title("Avg. MEG after artifact removal")
    textstr1 = 'Performance: %f' \
                %(calc_performance(evoked, evoked_after_ica))
    ax2.text(0.05, 0.95, textstr1, transform=ax2.transAxes, fontsize=14, 
             verticalalignment='top', bbox=props)
    
    # Plotting ecg signal again
    pl.plot(evoked.times*1e3, rescale_artifact(evoked, evoked_ecg), 'r', axes=ax1)
    
    # Plotting EOG signal before ICA
    eog_eve = mne.preprocessing.find_eog_events(raw, event_id_eog, ch_name=eog_ch_name)
    epochs = mne.Epochs(raw, eog_eve, event_id_eog, tmin, tmax, picks=picks_meg)
    evoked = epochs.average()

    ax3 = pl.subplot(222)
    evoked.plot(axes=ax3, xlim=xlim1)
    ylim_eog = dict(mag=ax3.get_ylim())
    pl.title("Avg. MEG showing artifacts")
    textstr1 = 'num_events=%d\nEpochs: tmin, tmax = %0.1f, %0.1f' %(len(eog_eve), tmin,tmax)
    ax3.text(0.05, 0.95, textstr1, transform=ax3.transAxes, fontsize=12, 
             verticalalignment='top', bbox=props)

    # Plotting eog signal
    picks_eog = mne.fiff.pick_channels(raw.info['ch_names'], 'EOG 002')
    epochs_eog = mne.Epochs(raw, eog_eve, event_id_eog, tmin, tmax, picks=picks_eog)
    evoked_eog = np.average(epochs_eog.get_data(), axis=0).flatten()*-1
    pl.plot(evoked.times*1e3, rescale_artifact(evoked, evoked_eog) * 10, 'r', axes=ax3)

    # Plotting signal after ICA
    epochs_after_ica = mne.Epochs(raw_cleaned, eog_eve, event_id_eog, tmin, tmax, picks=picks_meg)
    evoked_after_ica = epochs_after_ica.average()
    ax4 = pl.subplot(224)
    evoked_after_ica.plot(axes=ax4, ylim=ylim_eog, xlim=xlim1)

    pl.title("Avg MEG after artifact removal")
    textstr1 = 'Performance: %f'\
                %(calc_performance(evoked, evoked_after_ica))
    ax4.text(0.05, 0.95, textstr1, transform=ax4.transAxes, fontsize=14, 
             verticalalignment='top', bbox=props)

    # Plotting eog signal again
    pl.plot(evoked.times*1e3, rescale_artifact(evoked, evoked_eog) * 10, 'r', axes=ax1)
    pl.tight_layout()
    pl.savefig('artifact_removal_%s.png'%(raw_fname), format='png')
    pl.close('Compare')

    # Difference signal calculation
    events = mne.find_events(raw, stim_channel=stim_ch_name)
    evoked_stim_before_ica = mne.Epochs(raw, events, event_id_stim, tmin, tmax, proj=False,
                                        picks=picks_meg, preload=False, reject=None).average()
    evoked_stim_after_ica = mne.Epochs(raw_cleaned, events, event_id_stim, tmin, tmax, proj=False,
                                       picks=picks_meg, preload=False, reject=None).average()

    difference_signal_plot(evoked_stim_before_ica, evoked_stim_after_ica, 
                           textstr_diff='Performance:%f'
                           %(calc_performance(evoked_stim_before_ica, evoked_stim_after_ica)), 
                           save_title='ecg_eog_artifact_removal.png')
