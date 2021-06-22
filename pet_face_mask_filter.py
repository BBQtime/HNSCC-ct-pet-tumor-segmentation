import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
from multiprocessing import Pool
import glob
import numpy as np
import path

#path_in = "/mnt/faststorage/jintao/HNSCC/hecktor2021_train/resampled/" #change to your train folder


def face_mask_filter(patient):
    #_ct.nii.gz

    print("processing patient no. ", patient)

    CT_path = path_in + patient +'_ct.nii.gz'
    PET_path = path_in + patient +'_pt.nii.gz'


    pet_img = sitk.ReadImage(PET_path)
    pet = sitk.GetArrayFromImage(pet_img)

    ct_img = sitk.ReadImage(CT_path)
    ct = sitk.GetArrayFromImage(ct_img)

    fm  = ct > 0

    PTfm = pet*fm

    out_file = path_in + patient +'_ptfx.nii.gz'

    img_corr = sitk.GetImageFromArray(PTfm)
    img_corr.CopyInformation(pet_img)
    sitk.WriteImage(img_corr, out_file)

if __name__ == "__main__":
    path_in  = path.resampled_path

    file_list = glob.glob(os.path.join(path_in, '*'))

    patient_names = sorted( list(set(os.path.basename(patient)[:7] for patient in file_list)))
    print("patient names", patient_names)

    # parameterlist = []
    # for index, _ in enumerate(patient_names):
    #     parameterlist.extend([[patient_names[index]]])
    # print(len(parameterlist))

    p = Pool(processes=8)              # start 8 worker processes
    #print(parameterlist)
    p.map(face_mask_filter, patient_names)
    p.close()