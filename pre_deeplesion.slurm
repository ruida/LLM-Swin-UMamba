#!/bin/bash
###
#SBATCH --time=72:00:00
#SBATCH --mem=128g
#SBATCH --job-nam="pre_deeplesion"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100x:1
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
&&  nnUNetv2_plan_and_preprocess -d 710 --verify_dataset_integrity 2>&1| tee ${PWD##*/}_pre_deeplesion.log
