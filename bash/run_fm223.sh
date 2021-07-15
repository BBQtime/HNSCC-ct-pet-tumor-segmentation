CUDA_VISIBLE_DEVICES=1 nnUNet_train -c 3d_fullres nnUNetTrainerV2_1000DA 223 2 --npz
CUDA_VISIBLE_DEVICES=1 nnUNet_train 3d_fullres nnUNetTrainerV2_1000DA 223 3 --npz
CUDA_VISIBLE_DEVICES=1 nnUNet_train 3d_fullres nnUNetTrainerV2_1000DA 223 4 --npz
