#!/bin/sh
#SBATCH --partition=general # Request partition. Default is 'general' 
#SBATCH --qos=short         # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=0:02:00      # Request run time (wall-clock). Default is 1 minute
#SBATCH --ntasks=1          # Request number of parallel tasks per job. Default is 1
#SBATCH --cpus-per-task=2   # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem=4096      # Request memory (MB) per node. Default is 1024MB (1GB). For multiple tasks, specify --mem-per-cpu instead
#SBATCH --mail-type=ALL     # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=slurm_%j.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=slurm_%j.err # Set name of error log. %j is the Slurm jobId
#SBATCH --array=1-5

/usr/bin/scontrol show job -d "$SLURM_JOB_ID"  # check sbatch directives are working
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

# Remaining job commands go below here. For example, to run a Matlab script named "matlab_script.m", uncomment:
module use /opt/insy/modulefiles # Use DAIC INSY software collection
module load miniconda      # Load miniconda
conda activate /tudelft.net/staff-umbrella/tunoMSc2023/codes/gpt_env/
#srun matlab < matlab_script.m # Computations should be started with 'srun'.
python /tudelft.net/staff-umbrella/tunoMSc2023/codes/test.py
# ion.py .py n.py s and adapt them to load the software that your job requires
#module use /opt/insy/modulefiles          # Use DAIC INSY software collection
#module load cuda/11.2 cudnn/11.2-8.1.1.33 # Load certain versions of cuda and cudnn 
#srun python my_program.py # Computations should be started with 'srun'. For example:

# Measure GPU usage of your job (result)
# /usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"ous"
