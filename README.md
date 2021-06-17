# PET/CT Head and Neck tumor auto-segmentation

This is a repository for competition of MICCAI 2021: HECKTOR - head and neck gross tumor segmentation(GTV).




## Prerequisites:

code for resample images to istropical 1mm grid:

```
git https://github.com/voreille/hecktor
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

## Filter PET by face mask
PET has a poor spatial contrast, and its blurry boundery may highlight areas where there is no tissue. We utilized a facial mask to filter the PET image to diminish the impact.
```
python pet_face_mask_filter.py
```



