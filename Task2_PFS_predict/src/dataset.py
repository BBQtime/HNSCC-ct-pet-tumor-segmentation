import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import torch

class HecktorDataset(Dataset):
    """A class for fetching data samples.

    Parameters
    ----------
    paths_to_samples : list
        A list wherein each element is a tuple with two (three) `pathlib.Path` objects for a single patient.
        The first one is the path to the CT image, the second one - to the PET image. If `mode == 'train'`, a path to
        a ground truth mask must be provided for each patient.
    transforms
        Transformations applied to each data sample.
    mode : str
        Must be `train` or `test`. If `train`, a ground truth mask is loaded using a path from `paths_to_samples` and
        added to a sample.
        If `test`, an additional information (an affine array), that describes the position of the image data
        in a reference space, is added to each data sample. Ground truth masks are not loaded in this mode.

    Returns
    -------
    dict
        A dictionary corresponding to a data sample.
        Keys:
            id : A patient's ID.
            input : A numpy array containing CT & PET images stacked along the last (4th) dimension.
            target : A numpy array containing a ground truth mask.
            affine : A numpy array with the position of the image data in a reference space (needed for resampling).
    """

    def __init__(self, paths_to_samples, PFS_df, transforms=None, mode='train'):
        self.paths_to_samples = paths_to_samples
        self.transforms = transforms
        self.PFS_df = PFS_df
        if mode not in ['train', 'test']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode
        if mode == 'train':
            self.num_of_seqs = len(paths_to_samples[0]) - 1
        else:
            self.num_of_seqs = len(paths_to_samples[0])

    def __len__(self):
        return len(self.paths_to_samples)

    def __getitem__(self, index):
        sample = dict()

        #id_ = self.paths_to_samples[index][0].parent.stem
        id_ = self.paths_to_samples[index][0].parts[-1][:7]
        sample['id'] = id_
        img = [self.read_data(self.paths_to_samples[index][i]) for i in range(self.num_of_seqs)]
        img = np.stack(img, axis=-1)
        sample['input'] = img

        if self.mode == 'train':
            mask = self.read_data(self.paths_to_samples[index][-1])
            mask = np.expand_dims(mask, axis=3)

            assert img.shape[:-1] == mask.shape[:-1], \
                f"Shape mismatch for the image with the shape {img.shape} and the mask with the shape {mask.shape}."

            sample['target'] = mask

        else:
            sample['affine'] = self.read_data(self.paths_to_samples[index][0], False).affine
        if self.transforms:
            sample = self.transforms(sample)
        # i - mean/ std
        #sample['PFS'] = torch.as_tensor((self.PFS_df[self.PFS_df['PatientID']==id_]['Progression free survival'].iloc[0] - 1223.8)/601.9, dtype=torch.float32)
        # clip by 0-3000/3000 : normalize to 0-1
        normalize = np.clip(self.PFS_df[self.PFS_df['PatientID']==id_]['Progression free survival'].iloc[0],0,3000)/3000
        sample['PFS'] = torch.as_tensor(normalize, dtype=torch.float32)
        sample['Progression'] = self.PFS_df[self.PFS_df['PatientID']==id_]['Progression'].iloc[0]
        # don't nomalize
        #sample['PFS'] = torch.as_tensor(self.PFS_df[self.PFS_df['PatientID']==id_]['Progression free survival'].iloc[0], dtype=torch.float32)
        return sample

    @staticmethod
    def read_data(path_to_nifti, return_numpy=True):
        """Read a NIfTI image. Return a numpy array (default) or `nibabel.nifti1.Nifti1Image` object"""
        if return_numpy:
            return nib.load(str(path_to_nifti)).get_fdata()
        return nib.load(str(path_to_nifti))




class HecktorDatasetTest(Dataset):
    """A class for fetching data samples.

    Parameters
    ----------
    paths_to_samples : list
        A list wherein each element is a tuple with two (three) `pathlib.Path` objects for a single patient.
        The first one is the path to the CT image, the second one - to the PET image. If `mode == 'train'`, a path to
        a ground truth mask must be provided for each patient.
    transforms
        Transformations applied to each data sample.
    mode : str
        Must be `train` or `test`. If `train`, a ground truth mask is loaded using a path from `paths_to_samples` and
        added to a sample.
        If `test`, an additional information (an affine array), that describes the position of the image data
        in a reference space, is added to each data sample. Ground truth masks are not loaded in this mode.

    Returns
    -------
    dict
        A dictionary corresponding to a data sample.
        Keys:
            id : A patient's ID.
            input : A numpy array containing CT & PET images stacked along the last (4th) dimension.
            target : A numpy array containing a ground truth mask.
            affine : A numpy array with the position of the image data in a reference space (needed for resampling).
    """

    def __init__(self, paths_to_samples,  transforms=None, mode='test'):
        self.paths_to_samples = paths_to_samples
        self.transforms = transforms
        #self.PFS_df = PFS_df
        if mode not in ['train', 'test']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode
        if mode == 'train':
            self.num_of_seqs = len(paths_to_samples[0]) - 1
        else:
            self.num_of_seqs = len(paths_to_samples[0])

    def __len__(self):
        return len(self.paths_to_samples)

    def __getitem__(self, index):
        sample = dict()

        #id_ = self.paths_to_samples[index][0].parent.stem
        id_ = self.paths_to_samples[index][0].parts[-1][:7]
        sample['id'] = id_
        img = [self.read_data(self.paths_to_samples[index][i]) for i in range(self.num_of_seqs)]
        img = np.stack(img, axis=-1)
        sample['input'] = img

        if self.mode == 'train':
            mask = self.read_data(self.paths_to_samples[index][-1])
            mask = np.expand_dims(mask, axis=3)

            assert img.shape[:-1] == mask.shape[:-1], \
                f"Shape mismatch for the image with the shape {img.shape} and the mask with the shape {mask.shape}."

            sample['target'] = mask

        else:
            sample['affine'] = self.read_data(self.paths_to_samples[index][0], False).affine
        if self.transforms:
            sample = self.transforms(sample)
        # i - mean/ std
        #sample['PFS'] = torch.as_tensor((self.PFS_df[self.PFS_df['PatientID']==id_]['Progression free survival'].iloc[0] - 1223.8)/601.9, dtype=torch.float32)
        # clip by 0-3000/3000 : normalize to 0-1
        #normalize = np.clip(self.PFS_df[self.PFS_df['PatientID']==id_]['Progression free survival'].iloc[0],0,3000)/3000
        #sample['PFS'] = torch.as_tensor(normalize, dtype=torch.float32)
        #sample['Progression'] = self.PFS_df[self.PFS_df['PatientID']==id_]['Progression'].iloc[0]
        # don't nomalize
        #sample['PFS'] = torch.as_tensor(self.PFS_df[self.PFS_df['PatientID']==id_]['Progression free survival'].iloc[0], dtype=torch.float32)
        return sample

    @staticmethod
    def read_data(path_to_nifti, return_numpy=True):
        """Read a NIfTI image. Return a numpy array (default) or `nibabel.nifti1.Nifti1Image` object"""
        if return_numpy:
            return nib.load(str(path_to_nifti)).get_fdata()
        return nib.load(str(path_to_nifti))