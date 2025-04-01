#!/bin/bash
#SBATCH --job-name=ssins_analysis
#SBATCH --array=0-10  # SLURM job array index value, running job 0 to job 10 here
#SBATCH --ntasks=1  # One task per each job in the job array.
#SBATCH --cpus-per-task=2  # Number of CPUs per task
#SBATCH --time=10:00:00  # Wall time per task
#SBATCH --partition=Main
#SBATCH --mem=100GB  # Memory required per task
#SBATCH --output=logs/ssins-analysis-%A-%a.out

#### Note ####
# This SBATCH script use the SLURM job array to pass the job "index"
# to the python script. See e.g. https://hpc.nmsu.edu/discovery/slurm/job-arrays/
##############

# Some Numpy/Scipy functions can make use of multiple CPU cores through 
# multi-threading. Set variables that control this behaviour to
# match the SLURM CPUs per task.
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=$ SLURM_CPUS_PER_TASK}

# Activate conda and the correct Python environment
source ~/miniforge3/bin/activate
conda activate katcali
echo "Using Python: $(which python)"

# Pass variable SLURM_ARRAY_TASK_ID, which is the job array index value
echo "Executing a command: multi-block-Analysis.py ${SLURM_ARRAY_TASK_ID}"
python multi-block-executable.py ${SLURM_ARRAY_TASK_ID}
