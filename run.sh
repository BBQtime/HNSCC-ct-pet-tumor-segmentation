nnUNet_plan_and_preprocess -t 222 --verify_dataset_integrity
CUDA_VISIBLE_DEVICES=0 nnUNet_train 3d_fullres nnUNetTrainerV2_1000DA 222 0 --npz
