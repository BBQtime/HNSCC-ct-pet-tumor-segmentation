import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
from multiprocessing import Pool
import glob
import numpy as np


path_in = "/mnt/faststorage/jintao/HNSCC/hecktor2021_train/resampled/" #change to your train folder



file_list = glob.glob(os.path.join(path_in, '*'))

patient_names = sorted( list(set(os.path.basename(pt)[:7] for pt in file_list)))
print("patient names", patient_names)

def face_mask_filter(pt):
    #_ct.nii.gz

    print("processing patient no. ", pt)

    CT_path = path_in + pt +'_ct.nii.gz'
    PET_path = path_in + pt +'_pt.nii.gz'


    pet_img = sitk.ReadImage(PET_path)
    pet = sitk.GetArrayFromImage(pet_img)

    ct_img = sitk.ReadImage(CT_path)
    ct = sitk.GetArrayFromImage(ct_img)

    fm  = ct > 0

    PTfm = pet*fm

    out_file = path_in + pt +'_ptfx.nii.gz'

    img_corr = sitk.GetImageFromArray(PTfm)
    img_corr.CopyInformation(pet_img)
    sitk.WriteImage(img_corr, out_file)


# parameterlist = []
# for index, _ in enumerate(patient_names):
#     parameterlist.extend([[patient_names[index]]])
# print(len(parameterlist))

p = Pool(processes=8)              # start 8 worker processes
#print(parameterlist)
p.map(face_mask_filter, patient_names)