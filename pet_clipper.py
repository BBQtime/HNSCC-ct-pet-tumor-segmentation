import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
from multiprocessing import Pool
import glob
import numpy as np
import argparse
import path


def pet_clipper(patient):

    print("processing patient no. ", patient)

    PET_path = path_in + patient +'_pt.nii.gz'

    pet_img = sitk.ReadImage(PET_path)
    pet = sitk.GetArrayFromImage(pet_img)

    if mode == 'quantile':
        ptc = np.clip(pet, pet.min(), np.quantile(pet, clip))
    elif mode == 'number':
        ptc = np.clip(pet, pet.min(), clip)

    out_file = path_in + patient +'_ptc.nii.gz'

    img_corr = sitk.GetImageFromArray(ptc)
    img_corr.CopyInformation(pet_img)
    sitk.WriteImage(img_corr, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default='quantile',
                            help="clip mode choose from \'quantile\' and \'number\'")
    parser.add_argument("--clip", type=float, default=0.98,
                            help="clip value")
    args = parser.parse_args()
    # path_in = "/mnt/faststorage/jintao/HNSCC/hecktor2021_train/resampled/" #change to your train folder
    path_in  = path.resampled_path
    clip = args.clip
    mode = args.mode


    print('working direcotroy is: ', path_in)
    file_list = glob.glob(os.path.join(path_in, '*'))

    patient_names = sorted( list(set(os.path.basename(patient)[:7] for patient in file_list)))
    print("patient names", patient_names)

    # parameterlist = []
    # for index, _ in enumerate(patient_names):
    #     parameterlist.extend([[patient_names[index]]])
    # print(len(parameterlist))


    p = Pool(processes=8)              # start 8 worker processes
    #print(parameterlist)
    p.map(pet_clipper, patient_names)
    p.close()