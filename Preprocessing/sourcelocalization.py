import mne, sys
from mne.fiff import Evoked

try:
    subject = sys.argv[1]
    trigger = sys.argv[2]#Get the trigger is stim or resp
except:
    print "Please run with input file provided. Exiting"
    sys.exit()
subjects_dir = '/home/qdong/freesurfer/subjects/'
subject_path = subjects_dir + subject#Set the data path of the subject

fname_evoked = subject_path + '/MEG/ave_' + subject + '_audi_cued-raw_cle_%s.fif' %trigger#MEG data
evoked = Evoked(fname_evoked, setno=0, baseline=(None, 0))
mri = subject_path + '/MEG/' + subject + '-trans.fif'#Set the path of coordinates trans data
src = subject_path + '/bem/' + subject + '-ico-4-src.fif'#Set the path of src including dipole locations and orientations
bem = subject_path + '/bem/' + subject + '-5120-5120-5120-bem-sol.fif'#Set the path of file including the triangulation and conducivity\
#information together with the BEM
fname_cov = subject_path + '/MEG/' + subject + '_emptyroom_cov.fif'#Empty room noise covariance
fwd = mne.make_forward_solution(evoked.info, mri=mri, src=src, bem=bem,
                                fname=None, meg=True, eeg=False, mindist=5.0,
                                n_jobs=2, overwrite=True)
fwd = mne.convert_forward_solution(fwd, surf_ori=True)
mne.write_forward_solution(subject_path+'/%s_%s-fwd.fif' %(subject, trigger), fwd, overwrite=True)

#############################################################################
#                    make_inverse_operator                                  #
#############################################################################
snr = 3.0
lambda2 = 1.0 / snr ** 2

noise_cov = mne.read_cov(fname_cov)

# regularize noise covariance

noise_cov = mne.read_cov(fname_cov)
noise_cov = mne.cov.regularize(noise_cov, evoked.info,
                               mag=0.05, proj=True)

# Restrict forward solution as necessary for MEG
forward_meg = mne.fiff.pick_types_forward(fwd, meg=True, eeg=False)
inverse_operator_meg = mne.minimum_norm.make_inverse_operator(evoked.info, forward_meg, noise_cov,
                                                               loose=0.2, depth=0.8)
mne.minimum_norm.write_inverse_operator(subject_path+'/%s_%s-inv.fif' %(subject, trigger), inverse_operator_meg)#Write inverse operator


##################################################################################################
#      make sourcetime course and morph it to the common brain space                             #
##################################################################################################
stc = mne.minimum_norm.apply_inverse(evoked, inverse_operator_meg, lambda2, "dSPM")
stc_avg = mne.morph_data(subject, 'fsaverage', stc, 5, smooth=5)
#stc_avg.plot()
stc_avg.save(subject_path+'/fsaverage_'+subject+trigger, ftype='stc')

brain = stc_avg.plot(surface='inflated', hemi='lh', subjects_dir=subjects_dir)
brain.scale_data_colormap(fmin=8, fmid=12, fmax=15, transparent=True)
brain.show_view('lateral')
###
#### use peak getter to move vizualization to the time point of the peak
vertno_max, time_idx = stc_avg.get_peak(hemi='lh', time_as_index=True)
brain.set_data_time_index(time_idx)
###
#### draw marker at maximum peaking vertex 
brain.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='blue', scale_factor=0.6)
brain.save_image(subject_path+'/fsaverage_%s_dSPM_map_%s_lh_cortex.tiff' %(subject, trigger))

#mne.setup_source_space(subject, fname=True, spacing='ico4', surface='white', overwrite=False)
#mne_make_movie --inv 101611_stim-inv.fif --tmin 20 --tmax 120 --meas MEG/ave_101611_audi_cued-raw_cle_withoutSSP_stim.fif --morph fsaverage --smooth 5 --lh --mov first --subject 101611 --sLORETA
#mne_make_movie --stcin Ref_fsaverage101611_stim-lh.stc --tmin 20 --tmax 120 --mov first_dSPM --subject fsaverage --smooth 5 --lh --spm --fthresh 5 --fmid 8 --fmax 12