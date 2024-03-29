import pathlib
import os
import nibabel as nib
import torch
import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
class Predictor:
    """
    A class for building a model predictions.

    Parameters
    ----------
    model : a subclass of `torch.nn.Module`
        A model used for prediction.
    path_to_model_weights : list of (`pathlib.Path` or str) or (`pathlib.Path` or str)
        A path to model weights. Provide a path and use `self.predict` to build predictions using a single model.
        Use a list of paths and `self.ensemble_predict` to get predictions for an ensemble (the same architecture but
        different weights).
    dataloaders : `torch.utils.data.DataLoader`
        A dataloader fetching test samples.
    output_transforms
        Transforms applied to outputs.
    path_to_save_pred : `pathlib.Path` or str
        A path to a directory to save predictions
    """

    def __init__(self,
                 model,
                 path_to_model_weights,  # list of paths or path
                 dataloader,
                 output_transforms=None,
                 path_to_save_pred='.',
                 device="cuda:0"):

        self.model = model
        self.path_to_model_weights = [pathlib.Path(p) for p in path_to_model_weights] \
            if isinstance(path_to_model_weights, list) else pathlib.Path(path_to_model_weights)

        self.dataloader = dataloader
        self.output_transforms = output_transforms
        self.path_to_save_pred = pathlib.Path(path_to_save_pred)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results = []
        self.CI = 0
    def predict(self):
        """Run inference for an single model"""

        if self.device.type == 'cpu':
            print(f'Run inference for a model on CPU')
        else:
            print(f'Run inference for a model'
                  f' on {torch.cuda.get_device_name(torch.cuda.current_device())}')

        # Check if the directory exists:
        if not os.path.exists(os.path.dirname(self.path_to_save_pred)):
            os.makedirs(self.path_to_save_pred, exist_ok=True)

        # Send model to device:
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load model weights:
        self.model = self._load_model_weights(self.model, self.path_to_model_weights)

        # Inference:
        with torch.no_grad():
            for sample in self.dataloader:
                input = sample['input']
                input = input.to(self.device)

                output = self.model(input)
                output = output.cpu()

                sample['output'] = output

                # apply output transforms, if any:
                if self.output_transforms:
                    sample = self.output_transforms(sample)

                # Save prediction:
                self.results.append(self._save_preds(sample))
        save_df = pd.DataFrame(self.results)

        print(f'Predictions have been saved in {self.path_to_save_pred}')
        #self.CI = concordance_index(save_df['true_PFS'], save_df['pred_PFS'], save_df['Progression'])
        #save_df['CI'] = self.CI
        #print(f'CI: {self.CI}')
        save_df.to_csv(self.path_to_save_pred)

    def ensemble_predict(self):
        """Run inference for an ensemble of models"""

        if self.device.type == 'cpu':
            print(f'Run inference for an ensemble of {len(self.path_to_model_weights)} models on CPU')
        else:
            print(f'Run inference for an ensemble of {len(self.path_to_model_weights)} models'
                  f' on {torch.cuda.get_device_name(torch.cuda.current_device())}')

        # Check if the directory exists:
        if not os.path.exists(self.path_to_save_pred):
            os.makedirs(self.path_to_save_pred, exist_ok=True)

        # Send model to device:
        self.model = self.model.to(self.device)
        self.model.eval()

        # Inference:
        with torch.no_grad():
            for sample in self.dataloader:
                input = sample['input']
                input = input.to(self.device)

                ensemble_output = 0
                for path in self.path_to_model_weights:
                    self.model = self._load_model_weights(self.model, path)
                    output = self.model(input)
                    output = output.cpu()

                    ensemble_output += output

                ensemble_output /= len(self.path_to_model_weights)
                sample['output'] = ensemble_output

                # apply (ensemble) output transforms, if any:
                if self.output_transforms:
                    sample = self.output_transforms(sample)

                # Save prediction:
                self.results.append(self._save_preds(sample))

        print(f'Predictions have been saved in {self.path_to_save_pred}')

    @staticmethod
    def _save_preds(sample):
        result = {}
        preds = sample['output']
        sample_id = sample['id'][0]
        result['PatientID'] = sample['id'][0]
        result['pred_PFS'] = - int(preds.item()*3000)

        #result['Progression'] = sample['Progression'].item()

        #result['true_PFS']  = int(sample['PFS'].item()*3000)
        #result['AE'] = np.abs(result['true_PFS'] - result['pred_PFS'])
    
        print(result)

        return result
        #affine = sample['affine'][0].numpy()
        #preds = nib.Nifti1Image(preds, affine=affine)
        #nib.save(preds, str(path_to_dir / (sample_id + '.nii.gz')))

    @staticmethod
    def _load_model_weights(model, path_to_model_weights):
        model_state_dict = torch.load(path_to_model_weights, map_location=lambda storage, loc: storage)
        try:
            model.load_state_dict(model_state_dict, strict=True)
        except RuntimeError:
            # if model was trained in parallel
            #from collections import OrderedDict
            #new_model_state_dict = OrderedDict()
            #for k, v in model_state_dict.items():
            #    k = k.replace('module.', '')
            #    new_model_state_dict[k] = v
            model.load_state_dict(model_state_dict['model_state_dict'], strict=True)
        return model
