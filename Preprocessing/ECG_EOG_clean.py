# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:28:46 2014

@author: Dong Qunxi
"""

"""
Identify ECG artifacts using Cross Trial Phase Statistics 
Identify EOG artifacts using Correlation analysis and remove them.
Identify Brain response using Cross Trial Phase Statistics
Author: Praveen, Qunxi
"""
print __doc__

import numpy as np
import matplotlib.pylab as pl
import mne, sys, os
from mne.viz import tight_layout
from mne.fiff import Raw
from mne.preprocessing import ICA, find_ecg_events
from ctps import compute_ctps
from ctps import plot_ctps_panel
from mne.viz import plot_evoked
#view_plots = False
#CTP_RES = True
try:
    subject = sys.argv[1]
    
except:
    print "Please run with input file provided. Exiting"
    sys.exit()
subjects_dir = '/home/qdong/data/'
subject_path = subjects_dir + subject#Set the data path of the subject
raw_fname = subject_path + '/MEG/%s_audi_cued-raw.fif' %subject
raw_basename = os.path.splitext(os.path.basename(raw_fname))[0]
raw = Raw(raw_fname, preload=True)

flow, fhigh = 1.0, 45.0
filter_type = 'butter'
filter_order = 4
n_jobs = 4

n_components=0.99
n_pca_components=1.0
max_pca_components=None

ecg_ch_name = 'ECG 001'
res_ch_name = 'STI 013'
sti_ch_name = 'STI 014'
eog_ch_name = 'EOG 002'

picks = mne.fiff.pick_types(raw.info, meg=True, exclude='bads')
#raw.filter(l_freq=1, h_freq=45, picks=picks, n_jobs=n_jobs)
raw.filter(flow, fhigh, picks=picks, n_jobs=n_jobs, method='iir',
           iir_params={'ftype': filter_type, 'order': filter_order})



ica = ICA(n_components=n_components, n_pca_components=n_pca_components, max_pca_components=max_pca_components,
          random_state=0)
ica.decompose_raw(raw, picks=picks, decim=3)


##################EOG 1st rejection####################################

eog_ch_idx = [raw.ch_names.index(eog_ch_name)]
raw.filter(picks=eog_ch_idx, l_freq=1, h_freq=10)
eog_scores = ica.find_sources_raw(raw, raw[eog_ch_idx][0])
eog_idx = np.where(np.abs(eog_scores) > 0.1)[0]
ica.exclude += list(eog_idx)
print '%s has been identified as eog component and excluded' %(eog_idx)

#####################ECG Reject#########################################

# Filter sources in the ecg range
ica_ecg = ica
add_ecg_from_raw = mne.fiff.pick_types(raw.info, meg=False, ecg=True, include=ecg_ch_name)
sources_ecg = ica_ecg.sources_as_raw(raw, picks=add_ecg_from_raw)
ecg_eve, _, _ = find_ecg_events(sources_ecg, 999, ch_name=ecg_ch_name)
# drop non-data channels (ICA sources are type misc)
picks = mne.fiff.pick_types(sources_ecg.info, meg=False, misc=True)
sources_ecg.filter(l_freq=8, h_freq=16, method='iir', n_jobs=4)

# Epochs at R peak onset, from ecg_eve.
ica_epochs_ecg = mne.Epochs(sources_ecg, ecg_eve, event_id=999, tmin=-0.5, tmax=0.5,
                        picks=picks, preload=True, proj=False)

# Compute phase values and statistics (significance values pK)
#phase_trial_ecg, pk_dyn_ecg, _ = compute_ctps(ica_epochs_ecg.get_data())
_ , pk_dyn_ecg, _ = compute_ctps(ica_epochs_ecg.get_data())

# Get kuiper maxima
pk_max = pk_dyn_ecg.max(axis=1)

# Select ICs which can be attributed to cardiac artifacts
reject_sources = pk_max > 0.20  # bool array
reject_idx_ecg = np.where(reject_sources)[0].tolist()  # indices
ica.exclude += reject_idx_ecg
print '%s has been identified as ecg components and excluded' %(reject_idx_ecg)


raw_cle = ica.pick_sources_raw(raw, n_pca_components=n_pca_components)
#if view_plots:
 #   import compare_raw_files
  #  compare_raw_files.compare_raw_files(raw, raw_cle)

#ica.save('ctps_cor_resp_013.fif')
raw_cle_fil = subject_path + '/MEG/%s_cle.fif' %(raw_basename)
raw_cle.save(raw_cle_fil, overwrite=True)#Get the cleaned raw data
import compare_raw_files
compare_raw_files.compare_raw_files(raw, raw_cle)
