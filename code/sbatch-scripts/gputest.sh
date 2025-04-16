#!/bin/bash
# gputest.sh 4/16/25 D.C.
# One-task sbatch script using a GPU but not Apptainer, runs gputest.py
#SBATCH -c 6                  # allocate 4 CPU cores
#SBATCH -G 1                  # use one GPU
# #SBATCH -C v100               # insist on a V100 GPU
# #SBATCH -C a100               # insist on an A100 GPU (not avail on partition gpu)
#SBATCH -t 0:10:00            # time limit 10 min (default is 1 hr)
#SBATCH -p gpu                # submit to partition gpu
# #SBATCH -p gpu,gpu-preempt    # submit to partition gpu or gpu-preempt (<2 hrs)
scontrol write batch_script $SLURM_JOB_ID -;echo # print this batch script to output
echo nodelist=$SLURM_JOB_NODELIST    # get list of nodes used
module purge                  # unload all modules
module load conda/latest
module load cuda/12.6         # need CUDA to use a GPU
conda activate gpu            # environment with NumPy, SciPy, and CuPy
python gputest.py
