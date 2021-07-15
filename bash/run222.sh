CUDA_VISIBLE_DEVICES=0 nnUNet_train -c 3d_fullres nnUNetTrainerV2_1000DA 222 1 --npz
CUDA_VISIBLE_DEVICES=0 nnUNet_train -c 3d_fullres nnUNetTrainerV2_1000DA 222 3 --npz
CUDA_VISIBLE_DEVICES=0 nnUNet_train 3d_fullres nnUNetTrainerV2_1000DA 222 4 --npz
