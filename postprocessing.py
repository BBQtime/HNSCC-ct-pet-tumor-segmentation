from multiprocessing import Pool
import glob
import numpy as np
import SimpleITK as sitk
from pathlib import Path

"""
Rescale the prediction that with softmax probability maximum value less than 0.5 to 1

"""
import os
base_dir = '/mnt/faststorage/jintao/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task228_hecktor_sine+pet/nnUNetTrainerV2__nnUNetPlansv2.1/'
pred_dir = base_dir +'test/'
post_threshold = base_dir + 'post_threshold/'

if not os.path.isdir(post_threshold):
    os.mkdir(post_threshold)
patient_list = sorted([f.name[:7] for f in Path(pred_dir).rglob("*.nii.gz")])

file_list = glob.glob(os.path.join(pred_dir, '*'))

#patient_list = ['CHUV001']

for p in patient_list:
    nii = os.path.join(pred_dir, p +'.nii.gz')
    npz = os.path.join(pred_dir, p +'.npz')
    new_npz_path = os.path.join(post_threshold, p)
    new_nii_path = os.path.join(post_threshold, p +'.nii.gz')  
    
    pred = sitk.ReadImage(nii)

    prob_fg = np.load(npz)['softmax'][1].astype('float32')
    prob_max = prob_fg.max()
    if prob_max< 0.5:
        prob_fg /= prob_max
        prob_bg = 1 - prob_fg
        print(prob_bg.shape)
        new_img = [prob_bg, prob_fg]

        new_pred = (prob_fg > 0.5 ).astype(np.int8)

        new_nii = sitk.GetImageFromArray(new_pred)

        new_nii.CopyInformation(pred)
        sitk.WriteImage(new_nii, new_nii_path) 

        np.savez_compressed(new_npz_path, softmax=new_img)
    
        print(p, 'finished.')
    #if np.sum(pred) < 10:
        #print(np.sum(pred))
        
        
    