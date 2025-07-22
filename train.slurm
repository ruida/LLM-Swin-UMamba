#!/bin/bash
###
#SBATCH --time=128:00:00
#SBATCH --mem=128g
#SBATCH --job-name="train_715"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,ALL
 

source myconda
conda activate swin_umamba

ml CUDA/11.8
ml gcc/9.2.0

export nnUNet_raw="/data/ruida/segmentation/Swin-UMamba/data/nnUNet_raw"
export nnUNet_preprocessed="/data/ruida/segmentation/Swin-UMamba/data/nnUNet_preprocessed"
export nnUNet_results="/data/ruida/segmentation/Swin-UMamba/data/nnUNet_results"


cd "/data/ruida/segmentation/Swin-UMamba" \
&& nnUNetv2_train 715 2d all -tr nnUNetTrainerSwinUMamba 2>&1| tee ${PWD##*/}_train_715.log
