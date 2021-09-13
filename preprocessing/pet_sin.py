import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
from multiprocessing import Pool
import glob
import numpy as np

import path


def pet_sin_transform(patient):

    print("processing patient no. ", patient)

    PET_path = path_in + patient +'_pt.nii.gz'

    pet_img = sitk.ReadImage(PET_path)
    pet = sitk.GetArrayFromImage(pet_img)


    ptsin = np.sin(pet)

    out_file = path_in + patient +'_ptsin.nii.gz'

    img_ = sitk.GetImageFromArray(ptsin)
    img_.CopyInformation(pet_img)
    sitk.WriteImage(img_, out_file)


if __name__ == "__main__":
    
    #path_in  = path.resampled_path
    for path_in in path.resampled_path:
        print('working direcotroy is: ', path_in)
        file_list = glob.glob(os.path.join(path_in, '*'))

        patient_names = sorted( list(set(os.path.basename(patient)[:7] for patient in file_list)))
        print("patient names", patient_names)

        # parameterlist = []
        # for index, _ in enumerate(patient_names):
        #     parameterlist.extend([[patient_names[index]]])
        # print(len(parameterlist))


        p = Pool(processes=32)              # start 8 worker processes
        #print(parameterlist)
        p.map(pet_sin_transform, patient_names)
        p.close()