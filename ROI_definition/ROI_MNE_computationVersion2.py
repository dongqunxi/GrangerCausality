import numpy as np
import mne, sys, os
from mne.datasets import sample
from mne.fiff import Raw, pick_types
from mne.minimum_norm import (apply_inverse, apply_inverse_epochs,
                              read_inverse_operator)
from mne.connectivity import seed_target_indices, spectral_connectivity
from array import array
try:
    subject = sys.argv[1]
    trigger = sys.argv[2]#Get the trigger is stim or resp
except:
    print "Please run with input file provided. Exiting"
    sys.exit()

res_ch_name = 'STI 013'
sti_ch_name = 'STI 014'

#Load cleaned raw data based on trigger information 
subjects_dir = '/home/qdong/freesurfer/subjects/'
subject_path = subjects_dir + subject#Set the data path of the subject
raw_fname = subject_path + '/MEG/raw_%s_audi_cued-raw_cle_%s.fif' %(subject, trigger)
raw_basename = os.path.splitext(os.path.basename(raw_fname))[0]
raw = Raw(raw_fname, preload=True)
#Make events based on trigger information
if trigger == 'resp':
    tmin, tmax = -0.3, 0.3
    events = mne.find_events(raw, stim_channel=res_ch_name)
elif trigger == 'stim':
    tmin, tmax = -0.2, 0.4
    events = mne.find_events(raw, stim_channel=sti_ch_name)
picks = mne.fiff.pick_types(raw.info, meg=True, exclude='bads')
epochs = mne.Epochs(raw, events, 1, tmin, tmax, proj=False, picks=picks, preload=True, reject=None)

mri = subject_path + '/MEG/' + subject + '-trans.fif'#Set the path of coordinates trans data
src = subject_path + '/bem/' + subject + '-ico-4-src.fif'#Set the path of src including dipole locations and orientations
bem = subject_path + '/bem/' + subject + '-5120-5120-5120-bem-sol.fif'#Set the path of file including the triangulation and conducivity\
#information together with the BEM
fname_cov = subject_path + '/MEG/' + subject + '_emptyroom_cov.fif'#Empty room noise covariance
noise_cov = mne.read_cov(fname_cov)

#Forward solution
fwd = mne.make_forward_solution(epochs.info, mri=mri, src=src, bem=bem,
                                fname=None, meg=True, eeg=False, mindist=5.0,
                                n_jobs=2, overwrite=True)
fwd = mne.convert_forward_solution(fwd, surf_ori=True)

#Make Inverse operator
noise_cov = mne.cov.regularize(noise_cov, epochs.info,
                               mag=0.05, proj=True)
forward_meg = mne.fiff.pick_types_forward(fwd, meg=True, eeg=False)
inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, forward_meg, noise_cov,
                                             loose=0.2, depth=0.8)

##Load ROIs
#method = "MNE"
#snr = 1.0  # use lower SNR for single epochs
#lambda2 = 1.0 / snr ** 2
#Ind = 0
##stcs=[]
#labels=[]
#stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
         #                   pick_ori="normal")
snr = 3.0
method = "MNE"
lambda2 = 1.0 / snr ** 2
evoked = epochs.average()
stc = apply_inverse(evoked, inverse_operator, lambda2, method,
                    pick_ori="normal")
stc_avg = mne.morph_data(subject, 'fsaverage', stc, 5, smooth=5)
from surfer import Brain
subject_id = "fsaverage"
hemi = "split"
#surf = "smoothwm"
surf = 'inflated'
brain = Brain(subject_id, hemi, surf)
import os, random 
list_dirs = os.walk(subject_path + '/func_labels/') 
src = inverse_operator['src']
dipoles = []
color = ['#990033', '#9900CC', '#FF6600', '#FF3333', '#00CC33']
for root, dirs, files in list_dirs: 
    for f in files: 
        label_fname = os.path.join(root, f) 
        label = mne.read_label(label_fname)
        #label.values.fill(1.0)
        #label_morph = label.morph(subject_from='fsaverage', subject_to=subject, smooth=5, 
         #                        n_jobs=1, copy=True)
        stc_label = stc_avg.in_label(label)
        src_pow = np.sum(stc_label.data ** 2, axis=1)
        if label.hemi == 'lh':
           seed_vertno = stc_label.vertno[0][np.argmax(src_pow)]#Get the max MNE value within each ROI
           brain.add_foci(seed_vertno, coords_as_verts=True, hemi='lh', color=random.choice(color), scale_factor=0.6)
        elif label.hemi == 'rh':
           seed_vertno = stc_label.vertno[1][np.argmax(src_pow)]
           brain.add_foci(seed_vertno, coords_as_verts=True, hemi='rh', color=random.choice(color), scale_factor=0.6)
        dipole=[label.name, seed_vertno, stc_label.data[np.argmax(src_pow)]]
        dipoles.append(dipole)
f = open(root+'dipoles.txt', 'w')
for item in dipoles:
    print>>f, item
f.close()
X = dipoles[6][2]
Y = dipoles[5][2]
mean_X = np.mean(X, axis=0)
mean_Y = np.mean(Y, axis=0)
Sxy = 0
Sxx, Syy = 0, 0
i = 0
while i < len(X):
    Sxy += (X[i] - mean_X) * (Y[i] - mean_Y)
    Sxx += (X[i] - mean_X) * (X[i] - mean_X)
    Syy += (Y[i] - mean_Y) * (Y[i] - mean_Y)
    i += 1
r = Sxy / (np.sqrt(Sxx) * np.sqrt(Syy))
