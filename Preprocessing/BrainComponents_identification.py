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
n_pca_components=1.0
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
if trigger == 'resp':
    add_from_raw = mne.fiff.pick_types(raw.info, meg=False, resp=True, exclude='bads')
    sources_add = ica.sources_as_raw(raw, picks=add_from_raw)
    events = mne.find_events(sources_add, stim_channel=res_ch_name)
    raw_basename += '_resp'
elif trigger == 'stim':
    add_from_raw = mne.fiff.pick_types(raw.info, meg=False, stim=True, exclude='bads')
    sources_add = ica.sources_as_raw(raw, picks=add_from_raw)
    events = mne.find_events(sources_add, stim_channel=sti_ch_name)
    raw_basename += '_stim'
else:
    print "Incorrect trigger."
    sys.exit()
  # drop non-data channels (ICA sources are type misc)
  #ica.n_pca_components=None
picks = mne.fiff.pick_types(sources_add.info, meg=False, misc=True, exclude='bads')

  #Compare different bandwith of ICA components: 2-4, 4-8, 8-12, 12-16, 16-20Hz
l_f=2
Brain_idx=[]#The index of ICA related with trigger channels
for i in [4, 8, 12, 16]:
    h_f = i
    if l_f != 2:
      sources_add = ica.sources_as_raw(raw, picks=add_from_raw)
    
    sources_add.filter(l_freq=l_f, h_freq=h_f, n_jobs=4)
    temp = l_f
    l_f = h_f
    # Epochs at R peak onset, from stim_eve.
    ica_epochs_events = mne.Epochs(sources_add, events, event_id=1, tmin=-0.3, tmax=0.3,
                        picks=picks, preload=True, proj=False)
  
    # Compute phase values and statistics (significance values pK)
    #phase_trial_ecg, pk_dyn_ecg, _ = compute_ctps(ica_epochs_ecg.get_data())
    _ , pk_dyn_stim, phase_trial = compute_ctps(ica_epochs_events.get_data())
  
    # Get kuiper maxima
    pk_max = pk_dyn_stim.max(axis=1)
    
    # Select ICs which can be attributed to cardiac artifacts
    Brain_sources = pk_max > 0.1  # bool array, get the prominient components related with trigger
    Brain_ind = np.where(Brain_sources)[0].tolist() # indices
    #skip the null idx related with response
    Brain_idx += (Brain_ind)#Get the obvious sources related
    if Brain_ind == []:
    #plot_phase = False
    #if plot_phase == False: 
      continue
  
Brain_idx = list(set(Brain_idx))
print '%s has been identified as Brain response' %(Brain_idx)

#Recompose raw_cle data include the response
raw_brain = ica.pick_sources_raw(raw, include=Brain_idx, n_pca_components=n_pca_components)
raw_brain.save(subject_path+'/MEG/raw_%s.fif' %(raw_basename), overwrite=True)#Get the raw data related with brain response

#========Compare the brain response between raw before and after ctps=====
if trigger == 'resp':
    tmin, tmax = -0.3, 0.3
    events = mne.find_events(raw, stim_channel=res_ch_name)
elif trigger == 'stim':
    tmin, tmax = -0.2, 0.4
    events = mne.find_events(raw, stim_channel=sti_ch_name)
picks_bef = mne.fiff.pick_types(raw.info, meg=True, exclude='bads')
picks_aft = mne.fiff.pick_types(raw_brain.info, meg=True, exclude='bads')
evoked_before_ctps = mne.Epochs(raw, events, 1, tmin, tmax, proj=False, picks=picks_bef,   preload=False, reject=None).average()
evoked_after_ctps = mne.Epochs(raw_brain, events, 1, tmin, tmax, proj=False, picks=picks_aft,   preload=False, reject=None).average()
evoked_after_ctps.save(subject_path+'/MEG/ave_%s.fif' %(raw_basename))
times = evoked_before_ctps.times
ax1 = pl.subplot(211)
pl.title('CTPS Before, %s' %(subject))
pl.plot(times*1e3, evoked_before_ctps.data.T, axes=ax1)
#pl.plot(times*1e3, evoked_before_ctps.get_data().squeeze().T, axes=ax1)
pl.axvline(0, color='k', label='stim onset')
pl.xlabel('Time (ms)')
pl.ylabel('Evoked magnetic fields (AU)')
ax2 = pl.subplot(212)
pl.title('CTPS After, %s' %(subject))
pl.plot(times*1e3, evoked_after_ctps.data.T, axes=ax2)
#pl.plot(times*1e3, evoked_after_ctps.get_data().squeeze().T, axes=ax2)
pl.tight_layout()
pl.axvline(0, color='k', label='stim onset')
pl.xlabel('Time (ms)')
pl.ylabel('Evoked magnetic fields (AU)')
pl.show()
pl.savefig(subject_path+'/MEG/%s Brain signals before and after ctps.png' %(raw_basename))

