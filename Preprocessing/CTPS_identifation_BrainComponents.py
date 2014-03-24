# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:42:55 2014

@author: imenb101
"""

import numpy as np
import matplotlib.pylab as pl
import mne, sys, os
from mne.viz import tight_layout
from mne.fiff import Raw
from mne.preprocessing import ICA
from ctps import compute_ctps
from ctps import plot_ctps_panel

try:
    subject = sys.argv[1]
    trigger = sys.argv[2]#Get the trigger is stim or resp
except:
    print "Please run with input file provided. Exiting"
    sys.exit()

res_ch_name = 'STI 013'
sti_ch_name = 'STI 014'
n_components=0.99
n_pca_components=None
max_pca_components=None

subjects_dir = '/home/qdong/data/'
subject_path = subjects_dir + subject#Set the data path of the subject
#raw_fname = subject_path + '/MEG/ssp_cleaned_%s_audi_cued-raw_cle.fif' %subject
raw_fname = subject_path + '/MEG/%s_audi_cued-raw_cle.fif' %subject
raw_basename = os.path.splitext(os.path.basename(raw_fname))[0]
raw = Raw(raw_fname, preload=True)
picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, eog=False, 
                            stim=False, exclude='bads')
ica = ICA(n_components=n_components, n_pca_components=n_pca_components,  max_pca_components=max_pca_components, random_state=0)
ica.decompose_raw(raw, picks=picks, decim=3)
if trigger == 'resp':#'1' represents the response channel
    add_from_raw = mne.fiff.pick_types(raw.info, meg=False, resp=True, exclude='bads')
    sources_add = ica.sources_as_raw(raw, picks=add_from_raw)
    events = mne.find_events(sources_add, stim_channel=res_ch_name)
    raw_basename += '_resp'
elif trigger == 'stim':#'0' represents the stimuli channel
    add_from_raw = mne.fiff.pick_types(raw.info, meg=False, stim=True, exclude='bads')
    sources_add = ica.sources_as_raw(raw, picks=add_from_raw)
    events = mne.find_events(sources_add, stim_channel=sti_ch_name)
    raw_basename += '_stim'
else:
    print "Please select the triger channel '1' for response channel or '0' for stimilus channel."
    sys.exit()
  # drop non-data channels (ICA sources are type misc)
  #ica.n_pca_components=None
picks = mne.fiff.pick_types(sources_add.info, meg=False, misc=True, exclude='bads')

  #Compare different bandwith of ICA components: 2-4, 4-8, 8-12, 12-16, 16-20Hz
l_f = 2
Brain_idx1=[]#The index of ICA related with trigger channels
axes_band = [221, 222, 223, 224]
ax_index = 0
for i in [4, 8, 12, 16]:
    h_f = i
    get_ylim = True
    if l_f != 2:
      get_ylim = False
      sources_add = ica.sources_as_raw(raw, picks=add_from_raw)

    #sources_add.filter(l_freq=l_f, h_freq=h_f, method='iir', n_jobs=4)
    sources_add.filter(l_freq=l_f, h_freq=h_f, n_jobs=4, method='iir')
    this_band = '%i-%iHz' % (l_f, h_f)
    temp = l_f
    l_f = h_f
    # Epochs at R peak onset, from stim_eve.
    ica_epochs_events = mne.Epochs(sources_add, events, event_id=1, tmin=-0.3, tmax=0.3,
                        picks=picks, preload=True, proj=False)
    x_length = len(ica_epochs_events.ch_names)
    # Compute phase values and statistics (significance values pK)
    #phase_trial_ecg, pk_dyn_ecg, _ = compute_ctps(ica_epochs_ecg.get_data())
    _ , pk_dyn_stim, phase_trial = compute_ctps(ica_epochs_events.get_data())

    # Get kuiper maxima
    pk_max = pk_dyn_stim.max(axis=1)
    
    Brain_sources = pk_max > 0.1  # bool array, get the prominient components related with trigger
    Brain_ind = np.where(Brain_sources)[0].tolist() # indices
    #skip the null idx related with response
    Brain_idx1 += (Brain_ind)#Get the obvious sources related
    #Plot the bar
    #ax = pl.subplot(axes_band[ax_index])
    #pk_max.plot(axes=ax_index, ylim=ylim_ecg, xlim=xlim1)
    pl.subplot(axes_band[ax_index])
    x_bar = np.arange(x_length)
    pl.bar(x_bar, pk_max)
    for x in Brain_ind:
        pl.bar(x, pk_max[x], facecolor='r')
    pl.axhline(0.1, color='k', label='threshod')
    pl.xlabel('%s' %this_band)
    pl.ylim(0, 0.5)
    ax_index += 1

pl.tight_layout()
pl.show()
#pl.savefig(subject_path+'/MEG/ctps_distribution_%s_%s_withoutSSP.png'%(subject, trigger)) 
pl.savefig(subject_path+'/MEG/ctps_distribution_%s_%s.png'%(subject, trigger))  
Brain_idx = list(set(Brain_idx1))
print '%s has been identified as trigger components' %(Brain_idx)

