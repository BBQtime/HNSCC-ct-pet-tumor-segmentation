nnUNet_plan_and_preprocess -t 223 --verify_dataset_integrity
CUDA_VISIBLE_DEVICES=1 nnUNet_train 3d_fullres nnUNetTrainerV2_1000DA 223 0 --npz
CUDA_VISIBLE_DEVICES=1 nnUNet_train 3d_fullres nnUNetTrainerV2_1000DA 223 1 --npz
CUDA_VISIBLE_DEVICES=1 nnUNet_train 3d_fullres nnUNetTrainerV2_1000DA 223 2 --npz
CUDA_VISIBLE_DEVICES=1 nnUNet_train 3d_fullres nnUNetTrainerV2_1000DA 223 3 --npz
CUDA_VISIBLE_DEVICES=1 nnUNet_train 3d_fullres nnUNetTrainerV2_1000DA 223 4 --npz