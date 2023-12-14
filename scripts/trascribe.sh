#!/bin/bash

#SBATCH --time=01:00:00   
#SBATCH --ntasks=1
#SBATCH --nodes=1   
#SBATCH --mem-per-cpu=8G 

#SBATCH -o /home/st392/code/MultiTaskAVSR/scripts/outputsNLP/%A_%x.out
#SBATCH -e /home/st392/code/MultiTaskAVSR/scripts/outputsNLP/Error_%A_%x.out
#SBATCH --chdir /fslhome/st392/code
#SBATCH -J transcribe
#SBATCH --mail-user=st392@byu.edu
#SBATCH --mail-type=FAIL

python MultiTaskAVSR/transcribe.py file_path=$1