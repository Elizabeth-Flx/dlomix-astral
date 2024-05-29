#!/bin/bash

#SBATCH -J astral
#SBATCH -o logs/sbatch_out/%x.%A_%a.%N.out
#SBATCH -e logs/sbatch_out/%x.%A_%a.%N.gerr
#SBATCH -D ./
#SBATCH --get-user-env

#SBATCH --partition=shared-gpu          # compms-cpu-small | shared-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --tasks-per-node=1

##SBATCH --mail-user=ge27buk@mytum.de
##SBATCH --mail-type=end
#SBATCH --export=NONE
#SBATCH --time=96:00:00
##SBATCH --array=1-100%3

export CUDA_VERSION=11.8
export CUDNN_VERSION=8.9.7.29

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda

source ~/miniconda3/etc/profile.d/conda.sh
conda activate astral
export HF_DATASETS_CACHE="/cmnfs/proj/prosit_astral"
export HF_DATASETS_CACHE="/cmnfs/proj/prosit_astral/datasets"

export HF_HOME='/cmnfs/proj/prosit/ptms/huggingface'
export HF_DATASETS_CACHE='/cmnfs/proj/prosit/ptms/huggingface/datasets'

python -u Train_model_intensity.py &> logs/$1.log
