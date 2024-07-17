#!/bin/bash

#SBATCH --time=72:00:00   
#SBATCH --ntasks=1
#SBATCH --nodes=1   
#SBATCH --mem-per-cpu=16G 

#SBATCH -o /home/st392/code/MultiTaskAVSR/scripts/outputsNLP/%A_%a_%x.out
#SBATCH -e /home/st392/code/MultiTaskAVSR/scripts/outputsNLP/Error_%A_%a_%x.out
#SBATCH --chdir /fslhome/st392/code
#SBATCH -J transcribe
#SBATCH --mail-user=st392@byu.edu
#SBATCH --mail-type=FAIL

# python MultiTaskAVSR/transcribe.py data.modality=audiovisual file_path=$1 pretrained_model_path=$2 index=${SLURM_ARRAY_TASK_ID+5000}
python MultiTaskAVSR/transcribe.py data.modality=audiovisual file_path=$1 pretrained_model_path=$2 index=${SLURM_ARRAY_TASK_ID}