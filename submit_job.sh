#$ -S /bin/bash

#$ -o test.out

#$ -e test.err

#$ -V # Import the user's environment variables

#$ -j y

#$ -cwd # Current working directory

# Job name
#$ -N test

#$ -l gpu=true,gpu_type=!(gtx1080ti|rtx2080ti|titanx)

# Memory per slot and total
#$ -l tmem=32G
#$ -l h_vmem=32G

# Runtime limit (hh:mm:ss)
#$ -l h_rt=24:00:00

## Start logging
start_full=$(date +%Y.%m.%d-%T)
start=$(date +%s)
echo "Script name       :  'submit_job.sh'"
echo "Script started at :  $start_full"
echo "Dates             :  $(date)"
echo "Hostname          :  $(hostname)"
cpus=$(lscpu | grep "^CPU(s):")
echo "CPUs              :  $cpus"

echo "--------------------------------------"

source /share/apps/source_files/cuda/cuda-11.8.source
conda activate test
python main.py