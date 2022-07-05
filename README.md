# PET/CT Head and Neck tumor auto-segmentation

This is a repository for competition of MICCAI 2021: HECKTOR - head and neck gross tumor volume(GTV) segmentation.
### PET Normalizations to Improve Deep Learning Auto-Segmentation of Head and Neck Tumors in 3D PET/CT and Various Attempts at Patient Outcome Prediction

For more details, please reference to: 

### Ren, J., Huynh, B. N., Groendahl, A. R., Tomic, O., Futsaether, C. M., & Korreman, S. S. (2021, September). PET Normalizations to Improve Deep Learning Auto-Segmentation of Head and Neck Tumors in 3D PET/CT. In 3D Head and Neck Tumor Segmentation in PET/CT Challenge (pp. 83-91). Springer, Cham.
```
@inproceedings{ren2021pet,
  title={PET Normalizations to Improve Deep Learning Auto-Segmentation of Head and Neck Tumors in 3D PET/CT},
  author={Ren, Jintao and Huynh, Bao-Ngoc and Groendahl, Aurora Rosvoll and Tomic, Oliver and Futsaether, Cecilia Marie and Korreman, Stine Sofia},
  booktitle={3D Head and Neck Tumor Segmentation in PET/CT Challenge},
  pages={83--91},
  year={2021},
  organization={Springer}
}
```

![](Figure2.png) 


## Prerequisites:

HECKTOR public code for resample images to istropical 1mm grid with bounding box(144x144x144):

```
git clone https://github.com/voreille/hecktor
cd hecktor/src/resampling/
python resample.py
```

We use nnUNet as the baseline model for the development of GTV multimodality segmentation network.
install  nnUNet:
```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

convert data:
We've provded a script to convert sampled image to nnUNet format. Please change the your file location for both `downloaded_data_dir` and `downloaded_data_dir_test`

if convert train set only:
```
python data_conversion.py
```

if conver both train and test set:
```
python data_conversion.py --test=True
```

## Evaluation
### Evaluation on 5-folds cross validation(CV)
Run evaluation on 5-folds CV using following command for task id XXX:

`nnUNet_find_best_configuration -m 3d_fullres  -t XXX --strict`
