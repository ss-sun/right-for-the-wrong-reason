#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-24:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --gres=gpu:0              # optionally type and number of gpus
#SBATCH --mem=20G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/mnt/qb/work/baumgartner/sun22/logs/arr_%A_%a.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/qb/work/baumgartner/sun22/logs/arr_%A_%a.err   # File to which STDERR will be written
#SBATCH --array=0,1,2,3,4   # array of cityscapes random seeds
#SBATCH --mail-type=FAIL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=<susu.sun@uni-tuebingen.de>  # Email to which notifications will be sent
# print info about current job
scontrol show job $SLURM_JOB_ID 

# insert your commands here


# print info about current job
echo "---------- JOB INFOS ------------"
scontrol show job $SLURM_JOB_ID 
echo -e "---------------------------------\n"


source /mnt/qb/home/baumgartner/sun22/.bashrc

cd /mnt/qb/work/baumgartner/sun22/official_projects/right-for-the-wrong-reason

# Next activate the conda environment 
conda activate right_for_wrong


# Run our code
echo "-------- PYTHON OUTPUT ----------"

python3 ./data/contaminate_data.py --contaminated_dataset chexpert --contaminated_class "Cardiomegaly" --contamination_type tag --contamination_scale ${SLURM_ARRAY_TASK_ID}

python3 ./data/contaminate_data.py --contaminated_dataset chexpert --contaminated_class "Cardiomegaly" --contamination_type hyperintensities --contamination_scale ${SLURM_ARRAY_TASK_ID}

python3 ./data/contaminate_data.py --contaminated_dataset chexpert --contaminated_class "Cardiomegaly" --contamination_type obstruction --contamination_scale ${SLURM_ARRAY_TASK_ID}

echo "---------------------------------"


# Deactivate environment again
conda deactivate

