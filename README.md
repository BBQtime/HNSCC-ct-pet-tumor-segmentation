# HNSCC-ct-pet-tumor-segmentation

This is a repository for MICCAI 2021: HECKTOR  head and neck gross tumor segmentation(GTV).




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

