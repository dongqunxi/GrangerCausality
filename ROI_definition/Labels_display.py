print __doc__

import mne, os, sys
from surfer import Brain
import random
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

subject_id = "fsaverage"
hemi = "both"
#surf = "smoothwm"
surf = 'inflated'
brain = Brain(subject_id, hemi, surf)
list_dirs = os.walk(subject_path + '/func_labels/') 
color = ['#990033', '#9900CC', '#FF6600', '#FF3333', '#00CC33']
for root, dirs, files in list_dirs: 
    for f in files: 
        label_fname = os.path.join(root, f) 
        label = mne.read_label(label_fname)
        #label.values.fill(1.0)
        #label_morph = label.morph(subject_from='fsaverage', subject_to=subject, smooth=5, 
         #                        n_jobs=1, copy=True)
        if label.hemi == 'lh':
           brain.add_label(label, color=random.choice(color))
        elif label.hemi == 'rh':
           brain.add_label(label, color=random.choice(color))
#brain.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='blue', scale_factor=0.6)
# If the label lives in the normal place in the subjects directory,
# you can plot it by just using the name
#ref_ROI_fname = '/home/qdong/freesurfer/subjects/fsaverage/label/lh.Auditory_82.label'
#ref_label = mne.read_label(ref_ROI_fname)
#brain.add_label(ref_label, color="blue")
#brain.add_label("ROI1", color="blue")
#brain.add_label("ROI2", color="red")
#brain.add_label("ROI3", color="green")
#brain.add_label("ROI4", color="blue")
#brain.add_label("ROI5", color="blue")
#brain.add_label("ROI6", color="blue")
#brain.add_label("ROI7", color="blue")
#brain.add_label("ROI8", color="blue")
#brain.add_label("ROI9", color="blue")
#brain.add_label("ROI10", color="blue")
#brain.add_label("ROI11", color="blue")
#brain.add_label("RefROI1", color="yellow")
#
brain.show_view("lateral")
brain.save_image('/home/qdong/freesurfer/subjects/101611/lh_ROI2.tiff')
