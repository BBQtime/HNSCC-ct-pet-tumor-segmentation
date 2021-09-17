#!/usr/bin/env python
# coding: utf-8

# ## Building a model for the MICCAI 2020 HEad and neCK TumOR Task2 [(HECKTOR)](https://www.aicrowd.com/challenges/miccai-2021-hecktor)

# In[1]:


import os
import sys
import pathlib
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import torch
from torch.utils.data import DataLoader
import pandas as pd
import os

sys.path.append('../src/')

import dataset
import transforms
import losses
import metrics
import trainer
import models
import pandas as pd
from data import utils

from torch import Tensor
from torch.nn import functional as F

from lifelines.utils import concordance_index


def CanberraMetric1(y_pred, y):
    
    errors = torch.abs(y - y_pred) / (torch.abs(y_pred) + torch.abs(y) + 1e-15)

    _sum_of_errors = torch.sum(errors)

    return _sum_of_errors
        
def MeanAbsoluteRelativeError(y_pred, y):
    return torch.mean(torch.abs(y_pred - y.view_as(y_pred)) / torch.abs(y.view_as(y_pred)))
    
class CanberraHuberLoss(torch.nn.Module):


    def __init__(self) -> None:
        super(CanberraHuberLoss, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return CanberraMetric1(input, target) + 2*F.huber_loss(input, target, reduction='mean', delta=1) 


import pickle
def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def main(fold):
    #hyper-parameters
    train_batch_size = 4
    val_batch_size = 1
    num_workers = 32  # for example, use a number of CPU cores for dataloader

    ## training hyper-parameters
    n_cls = 1  # number of classes
    lr = 3e-6 # initial learning rate
    n_epochs = 150  # number of epochs

    train_val_split = load_pickle('/home/sysgen/gitlab/HNSCC-ct-pet-tumor-segmentation/task2_PFS_predict/src/data/nnUNet_splits/splits_final.pkl')
    path_to_data = pathlib.Path('/mnt/faststorage/jintao/HNSCC/hecktor2021_train/resampled/')
    all_paths = utils.get_paths_to_patient_files(path_to_imgs=path_to_data, append_mask=True)

    print(f'Total number of patients: {len(all_paths)}')

    path_to_pkl = '/mnt/faststorage/jintao/nnUNet/nnUNet_preprocessed/Task229_hecktor_base_focal/splits_final.pkl'
    path_to_PFS_csv = pathlib.Path('/mnt/faststorage/jintao/HNSCC/hecktor2021_train/hecktor2021_patient_endpoint_training.csv')
    PFS_df = pd.read_csv(path_to_PFS_csv)

    #split
    train_paths, val_paths = utils.get_nnUnet_train_val_paths(all_paths=all_paths, path_to_train_val_pkl=path_to_pkl, fold=fold)


    print(f'Total number of patients in TRAIN fold: \t{len(train_paths)}')
    print(f'Total number of patients in VAL fold: \t\t{len(val_paths)}')

    #Augmentation
    train_transforms = transforms.Compose([
        transforms.RandomRotation(p=0.5, angle_range=[0, 20]),
        transforms.NormalizeIntensity(),
        transforms.ToTensor()
    ])


    val_transforms = transforms.Compose([
        transforms.NormalizeIntensity(),
        transforms.ToTensor()
    ])


    # Datasets:
    train_set = dataset.HecktorDataset(train_paths, PFS_df, transforms=train_transforms)
    val_set = dataset.HecktorDataset(val_paths, PFS_df, transforms=val_transforms)

    # Dataloaders:
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
        
    }

    train_sample = next(iter(train_loader))
    print(f'Patient: \t{train_sample["id"]}')
    print(f'Input: \t\t{train_sample["input"].size()}')
    print(f'Target: \t{train_sample["target"].size()}')
    print(f'Target: \t{train_sample["PFS"].size()}')

    #from lifelines.utils import concordance_index
    #from ignite.contrib.metrics.regression import CanberraMetric
    #from livelossplot import PlotLosses

    model = models.FastSmoothSENormDeepEncoder_supervision_skip_no_drop(in_channels=2, n_cls=1, n_filters=24)
    #model = models.Encoder(in_channels=3, n_cls=1, n_filters=64)
    #criterion = torch.nn.MSELoss()
    #criterion = torch.nn.HuberLoss()
    criterion = CanberraHuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    #T_0= 25  # parameter for 'torch.optim.lr_scheduler.CosineAnnealingWarmRestarts'
    #eta_min= 1e-8  # parameter for 'torch.optim.lr_scheduler.CosineAnnealingWarmRestarts'
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, eta_min=eta_min)

    metric =  CanberraMetric1 #MeanAbsoluteRelativeError #concordance_index

    print('Total number of parameters:', get_n_params(model))


    trainer_ = trainer.ModelTrainer(model=model,
                                    dataloaders=dataloaders,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    metric=metric,
                                    scheduler=None,
                                    num_epochs=n_epochs,
                                    save_last_model=True,
                                    livelossplot = False)
    trainer_.train_model()

    path_to_save_dir = pathlib.Path('/home/sysgen/gitlab/HNSCC-ct-pet-tumor-segmentation/task2_PFS_predict/results/fold-'+str(fold)+'/')
    trainer_.save_results(path_to_save_dir)
    summary = pd.read_csv(path_to_save_dir / 'summary.csv')
    
    #predict.py -p ../config/model_predict.yaml -f 0

if __name__ == "__main__":
    for i in range(5):
        print(f"Training fold {i}")
        main(i)