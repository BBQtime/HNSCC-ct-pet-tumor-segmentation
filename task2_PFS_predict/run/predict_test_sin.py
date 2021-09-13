import sys
import argparse
import yaml
import pathlib
import os
import torch
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True

sys.path.append('../src/')
sys.path.append('../src/data/')
sys.path.append('../config/')
import dataset
import transforms
import utils
import models
import predictor
import pandas as pd

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    fold = str(args.fold)
    path_to_config = pathlib.Path('../config/model_predict.yaml')

    with open(path_to_config) as f:
        config = yaml.safe_load(f)

    # read config:
    path_to_test_data = pathlib.Path(config['path_to_test_data'])
    path_to_save_dir = config['path_to_save_ch_sin_dir']
    num_workers = int(config['num_workers'])
    n_cls = int(config['n_cls'])
    in_channels = int(config['in_channels'])
    n_filters = int(config['n_filters'])
    reduction = int(config['reduction'])

    print('args.best:', args.best)
    if args.weight !=None:
        path_to_weights = args.weight
    else: 
        if args.best== True:
            path_to_weights = os.path.join(path_to_save_dir, 'fold-'+fold, 'best_model_weights.pt')
        else:
            path_to_weights = os.path.join(path_to_save_dir, 'fold-'+fold, 'last_model_checkpoint.tar')
    print("Weight:",path_to_weights)

    # test data paths:
    all_paths = sorted(utils.get_paths_to_patient_files(path_to_imgs=path_to_test_data, append_mask=False, get_sin=True))

    # input transforms:
    input_transforms = transforms.Compose([
        transforms.NormalizeIntensity(),
        transforms.ToTensor(mode='test')
    ])

    path_to_pkl = '/mnt/faststorage/jintao/nnUNet/nnUNet_preprocessed/Task229_hecktor_base_focal/splits_final.pkl'

    #train_paths, val_paths = utils.get_nnUnet_train_val_paths(all_paths=all_paths, path_to_train_val_pkl=path_to_pkl, fold=args.fold)

    #print(list(set([p[:7] for p in os.listdir(path_to_imgs) if os.path.isfile(path_to_imgs / p)])))
    """   
    # ! change value from normalized back to PFS days.
    
    # ensemble output transforms:
    output_transforms = [
        transforms.InverseToTensor(),
        transforms.CheckOutputShape(shape=(144, 144, 144))
    ]
    if not probs:
        output_transforms.append(transforms.ProbsToLabels())

    output_transforms = transforms.Compose(output_transforms)
    """
    # dataset and dataloader:
    #path_to_PFS_csv = pathlib.Path('/mnt/faststorage/jintao/HNSCC/hecktor2021_train/hecktor2021_patient_endpoint_training.csv')
    #PFS_df = pd.read_csv(path_to_PFS_csv)
    test_paths = all_paths
    data_set = dataset.HecktorDatasetTest(test_paths, transforms=input_transforms, mode='test')
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=num_workers)

    # model:
    model = models.FastSmoothSENormDeepEncoder_supervision_skip_no_drop(in_channels=in_channels, n_cls=n_cls, n_filters=n_filters)

    # init predictor:
    path_to_save_pred = path_to_save_dir+'fold-'+fold+'/test.csv'
    predictor_ = predictor.Predictor(
        model=model,
        path_to_model_weights=path_to_weights,
        dataloader=data_loader,
        output_transforms=None,
        path_to_save_pred=path_to_save_pred
    )

    # check if multiple paths were provided to run an ensemble:
    if isinstance(path_to_weights, list):
        predictor_.ensemble_predict()

    elif isinstance(path_to_weights, str):
        predictor_.predict()

    else:
        raise ValueError(f"Argument 'path_to_weights' must be str or list of str, provided {type(path_to_weights)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Inference Script')
    parser.add_argument("-w", "--weight", type=str, required=False, default=None, help="path to the weight")
    parser.add_argument("-b", "--best", type=str2bool, required=False, default=False, help="if load best weight")
    parser.add_argument("-f", "--fold", type=int, required=True, help="fold to predict")
    args = parser.parse_args()
    main(args)
