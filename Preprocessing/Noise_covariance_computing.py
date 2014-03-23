"""
Input parameter: <subject>, the name of subject
Process the files: <subject>_emptyroom.fif
Compute the noise matrix
Author: Praveen, Qunxi
"""
import mne, sys, os
import pylab as pl
import numpy as np

flow, fhigh = 1.0, 45.0
filter_type = 'butter'
filter_order = 4
njobs = 4

try:
    subject = sys.argv[1]#Get the subject
except:
    print "Please run with input file provided. Exiting"
    sys.exit()
subjects_dir = '/home/qdong/freesurfer/subjects/'
subject_path = subjects_dir + subject#Set the data path of the subject

raw_empty_fname = subject_path + '/MEG/%s_emptyroom.fif' %subject
raw_empty = mne.fiff.Raw(raw_empty_fname, preload=True)
#Filter the empty room data
picks_empty = mne.fiff.pick_types(raw_empty.info, meg=True, eeg=False, eog=True, ecg=True, stim=False, exclude='bads') 
raw_empty.filter(flow, fhigh, picks=picks_empty, n_jobs=njobs, method='iir', 
           iir_params={'ftype': filter_type, 'order': filter_order})          
#Get the basename
raw_empty_basename = os.path.splitext(os.path.basename(raw_empty_fname))[0]
cov = mne.compute_raw_data_covariance(raw_empty, picks=picks_empty)
mne.write_cov(subject_path+'/MEG/%s_cov.fif' %(raw_empty_basename), cov)