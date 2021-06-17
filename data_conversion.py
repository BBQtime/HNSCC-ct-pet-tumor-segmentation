#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
from collections import OrderedDict
import os
import glob
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
import SimpleITK as sitk
import shutil
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

                    
"""
data conversion of hector 2021 for nnUNet support
"""

if __name__ == "__main__":
    """
    REMEMBER TO CONVERT LABELS BACK TO BRATS CONVENTION AFTER PREDICTION!
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("test", type=str2bool, nargs='?',
                            const=True, default=False,
                            help="also convert test set")

    args = parser.parse_args()

    task_name = "Task222_hecktor"
    downloaded_data_dir = "/mnt/faststorage/jintao/HNSCC/hecktor2021_train/resampled/"

    print('working on ', downloaded_data_dir)
    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")
    #target_imagesVal = join(target_base, "imagesVal")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")


    maybe_mkdir_p(target_imagesTr)
    #maybe_mkdir_p(target_imagesVal)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)
    maybe_mkdir_p(target_labelsTs)

    file_list = glob.glob(os.path.join(downloaded_data_dir, '*'))

    patient_names = sorted( list(set(os.path.basename(pt)[:7] for pt in file_list)))
    print("patient names", patient_names)

    cur = downloaded_data_dir

    for p in patient_names:
        patient_name = p

        ct = os.path.join(downloaded_data_dir, p+"_ct.nii.gz")
        pt = os.path.join(downloaded_data_dir, p+"_pt.nii.gz")
        gtv = os.path.join(downloaded_data_dir, p+"_gtvt.nii.gz")

        print(ct)
        assert all([
            isfile(ct),
            isfile(pt),
            isfile(gtv)
        ]), "%s" % patient_name

        shutil.copy(ct, join(target_imagesTr, patient_name + "_0000.nii.gz"))
        shutil.copy(pt, join(target_imagesTr, patient_name + "_0001.nii.gz"))
        shutil.copy(gtv, join(target_labelsTr, patient_name + ".nii.gz"))

    if args.test == True:
        downloaded_data_dir_test = "/mnt/faststorage/jintao/HNSCC/hecktor2021_test/resampled/"

        test_patient_names = []
        test_file_list = glob.glob(os.path.join(downloaded_data_dir, '*'))

        test_patient_names = sorted( list(set(os.path.basename(pt)[:7] for pt in test_file_list)))
        print("patient names", patient_names)

        cur = downloaded_data_dir_test
        for p in test_patient_names:
            patient_name = p
            ct = os.path.join(downloaded_data_dir,p+"_ct.nii.gz")
            pt = os.path.join(downloaded_data_dir,p+"_pt.nii.gz")
            gtv = os.path.join(downloaded_data_dir,p+"_gtvt.nii.gz")

            assert all([
                isfile(ct),
                isfile(pt),
                isfile(gtv)
            ]), "%s" % patient_name

            shutil.copy(ct, join(target_imagesTs, patient_name + "_0000.nii.gz"))
            shutil.copy(pt, join(target_imagesTs, patient_name + "_0001.nii.gz"))
            shutil.copy(gtv, join(target_labelsTs, patient_name + ".nii.gz"))

    json_dict = OrderedDict()
    json_dict['name'] = "Task222_hecktor"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "NA"
    json_dict['licence'] = "NA"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
        "1": "PT"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "GTVt"
    }
    json_dict['numTraining'] = len(patient_names)
    if args.test == True:
        json_dict['numTest'] = len(test_patient_names)

    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                            patient_names]

    if args.test == True:

        json_dict['test'] =  [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in
                            test_patient_names]

    save_json(json_dict, join(target_base, "dataset.json"))