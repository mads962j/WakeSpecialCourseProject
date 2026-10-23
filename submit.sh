#!/bin/bash
#BSUB -J UncertaintyTogglingWake
#BSUB -q hpc
#BSUB -W 01:00
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -o ./logs_lsf/UncertaintyTogglingWake%J.out
#BSUB -e ./logs_lsf/UncertaintyTogglingWake%J.err
#BSUB -B
#BSUB -N
#BSUB -n 1
#BSUB -R "span[hosts=1]"

#Activate the conda environment
source /zhome/d6/f/230449/courses/wake_dynamics/pywake_ellipsys/environments/PyWake_EllipSys/bin/activate

# Run the Python script with the specified arguments
time python UncertaintyTogglingWake.py
